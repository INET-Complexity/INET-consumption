from dataclasses import is_dataclass, replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import macromodel.agents.households.households as households_module
from macro_data.readers.emission_fraction.emission_fraction_reader import EmissionFractions
from macromodel.agents.households.func.borrow_vs_sell import (
    PREFERRED_MARGIN_BORROW,
    PREFERRED_MARGIN_SELL,
)
from macromodel.agents.households.func.consumer_distress import CURRENT, DELINQUENT, FICP
from macromodel.agents.households.func.consumption import CreditAugmentedConsumption
from macromodel.agents.households.func.financial_feasibility import PostGrantFeasiblePlan, PreGrantFeasiblePlan
from macromodel.agents.households.func.portfolio_diagnostics import Stage4HouseholdDiagnostics
from macromodel.agents.households.func.portfolio_rebalancing import PortfolioRebalancingResult
from macromodel.agents.households.func.wealth import PaperAssetReturnWealthSetter
from macromodel.agents.households.income_belief_learning import compute_zeta
from macromodel.configurations.households_configuration import HouseholdsConfiguration


def _replace_post_grant_plan(households, **updates):
    plan = households.post_grant_feasible_plan
    if is_dataclass(plan):
        households.post_grant_feasible_plan = replace(plan, **updates)
        return
    for name, value in updates.items():
        setattr(plan, name, value)


def _setup_emission_ts(households, n_hh, n_industries):
    """Inject a non-NaN spending matrix and emission timeseries keys."""
    households.ts.dicts["nominal_amount_spent_in_lcu"][-1] = np.full((n_hh, n_industries), 10.0)
    for key in [
        "consumption_emissions",
        "investment_emissions",
        "consumption_emissions_by_good",
        "investment_emissions_by_good",
        "coal_consumption_emissions",
        "oil_consumption_emissions",
        "gas_consumption_emissions",
        "refined_products_consumption_emissions",
        "coal_investment_emissions",
        "oil_investment_emissions",
        "gas_investment_emissions",
        "refined_products_investment_emissions",
    ]:
        size = n_industries if "by_good" in key else n_hh
        households.ts.dicts[key] = [np.zeros(size)]


class TestHouseholds:
    def test__create(self, test_households):
        assert test_households.country_name == "FRA"
        assert test_households.uses_feasibility_resolver is True
        assert test_households.pre_grant_feasible_plan is None
        assert test_households.post_grant_feasible_plan is None
        for field_name in [
            "consumption_before_floor",
            "residual_shortfall_before_floor",
            "consumption_after_floor",
            "consumption_cut_amount",
            "remaining_subsistence_shortfall",
            "floor_binding",
            "stage5_subsistence_support",
            "consumer_payment_suspension_needed",
            "consumer_payment_suspension_amount",
            "mortgage_payment_suspension_needed",
            "mortgage_payment_suspension_amount",
            "scheduled_consumer_payment",
            "actual_consumer_payment",
            "consumer_payment_missed",
            "missed_payment_count_consumer",
            "consumer_distress_state",
            "ficp_state",
            "ficp_exclusion_remaining_periods",
        ]:
            assert field_name in test_households.ts.get_keys()

    def test__handle_insolvency_supports_legacy_handler_signature(self, test_households, monkeypatch):
        class LegacyHandler:
            def handle_insolvency(self, households, banks, credit_market):
                return 1.0, 2.0, 3.0

        monkeypatch.setitem(test_households.functions, "insolvency", LegacyHandler())

        assert test_households.handle_insolvency(None, None, consumer_terminal_removal_exclusion=np.zeros((1, 1))) == (
            1.0,
            2.0,
            3.0,
        )

    def test__received_consumption_loans_starts_unsettled(self, test_households):
        received_consumption_loans = np.asarray(test_households.ts.current("received_consumption_loans"), dtype=float)

        assert np.isnan(received_consumption_loans).all()

    def test__households_states(self, test_households):
        assert test_households is not None
        for state in [
            "saving_rates_model",
            "social_transfers_model",
            "average_saving_rate",
            "Type",
            "Corresponding Bank ID",
            "Corresponding Inhabited House ID",
            "Corresponding Property Owner",
            "Tenure Status of the Main Residence",
            "corr_individuals",
            "corr_renters",
            "Consumption Units",
            "household_head_age",
            "household_members_in_employment",
            "population_scale_factor",
        ]:
            assert state in test_households.states.keys()

    def test__household_head_age_and_employment_states_are_populated_for_every_household(
        self, test_households, datawrapper
    ):
        # Integration smoke check against the real FRA population: every
        # household gets a finite head age and a non-negative employment
        # count, and the count never exceeds the household's own member
        # count. The head-selection rule itself (reference-person-first,
        # oldest-member fallback) is independently unit-tested against a
        # hand-built fixture in test_portfolio_target_share.py's
        # compute_household_head_covariates tests, not re-derived here.
        individual_data = datawrapper.synthetic_countries["FRA"].population.individual_data
        n_individuals = len(individual_data)

        for hh_id, corr_individuals in enumerate(test_households.states["corr_individuals"]):
            member_ids = np.asarray(corr_individuals, dtype=int)
            assert member_ids.size > 0
            assert np.all((member_ids >= 0) & (member_ids < n_individuals))
            assert 0 <= test_households.states["household_members_in_employment"][hh_id] <= member_ids.size

    def test__household_head_age_and_employment_count_are_finite_for_every_household(self, test_households):
        # Every household has at least one member by construction, so neither
        # the reference-person branch nor the oldest-member fallback in the
        # production code should ever leave a NaN/inf behind.
        assert np.all(np.isfinite(test_households.states["household_head_age"]))
        assert np.all(np.isfinite(test_households.states["household_members_in_employment"]))
        assert np.all(test_households.states["household_members_in_employment"] >= 0)

    def test__population_scale_factor_is_a_positive_scalar(self, test_households):
        assert test_households.states["population_scale_factor"] > 0.0

    def test__consumption_units_refresh_only_when_age_band_composition_changes(self, test_households):
        max_individual_id = max(
            max(corr_individuals) for corr_individuals in test_households.states["corr_individuals"]
        )
        individual_ages = np.full(max_individual_id + 1, 30.0)
        test_households.states["corr_individuals"][0] = np.array([0, 1, 2])
        individual_ages[[0, 1, 2]] = [30.0, 13.0, 10.0]

        test_households.mark_consumption_units_dirty()
        assert test_households.refresh_consumption_units_if_needed(individual_ages)
        assert test_households.states["Consumption Units"][0] == 1.6

        individual_ages[0] = 31.0
        assert not test_households.refresh_consumption_units_if_needed(individual_ages)
        assert test_households.states["Consumption Units"][0] == 1.6

        individual_ages[1] = 14.0
        assert test_households.refresh_consumption_units_if_needed(individual_ages)
        assert test_households.states["Consumption Units"][0] == 1.8

    def test__initial_main_residence_wealth_matches_owner_tenure_state(self, test_households):
        owns_main_residence = np.isin(test_households.states["Tenure Status of the Main Residence"], [1, 2, 4]) & (
            test_households.states["Corresponding Inhabited House ID"] != -1
        )

        assert np.all(test_households.ts.initial("wealth_main_residence")[~owns_main_residence] == 0.0)
        np.testing.assert_allclose(
            test_households.ts.initial("wealth_real_assets"),
            test_households.ts.initial("wealth_main_residence")
            + test_households.ts.initial("wealth_other_properties")
            + test_households.ts.initial("wealth_other_real_assets"),
        )
        np.testing.assert_allclose(
            test_households.ts.initial("wealth"),
            test_households.ts.initial("wealth_real_assets") + test_households.ts.initial("wealth_financial_assets"),
        )
        np.testing.assert_allclose(
            test_households.ts.initial("net_wealth"),
            test_households.ts.initial("wealth") - test_households.ts.initial("debt"),
        )

    def test__mortgage_target_ignores_rentals(self, test_households):
        n_households = test_households.ts.current("n_households")

        # Make the mortgage target depend only on target_house_price.
        test_households.ts.override_current(
            "target_consumption",
            np.zeros_like(test_households.ts.current("target_consumption")),
        )
        test_households.ts.override_current(
            "expected_income",
            np.zeros_like(test_households.ts.current("expected_income")),
        )
        test_households.ts.override_current(
            "rent",
            np.zeros_like(test_households.ts.current("rent")),
        )
        test_households.ts.override_current(
            "wealth_financial_assets",
            np.zeros_like(test_households.ts.current("wealth_financial_assets")),
        )
        test_households.pre_grant_feasible_plan = PreGrantFeasiblePlan(
            liquidity_shortfall_before_repair=np.zeros(n_households),
            funded_from_liquid_assets=np.zeros(n_households),
            residual_shortfall_after_lfa=np.zeros(n_households),
            credit_requested=np.zeros(n_households),
            planned_liquidation_total=np.zeros(n_households),
        )

        current_sales = pd.DataFrame(
            {
                "sales_types": ["Rental", "Sell"],
                "buyer_id": [0, 1],
                "price_or_rent": [100.0, 100000.0],
            }
        )
        test_households.compute_target_credit(current_sales=current_sales)

        target_mortgage = test_households.ts.current("target_mortgage")
        assert target_mortgage.shape == (n_households,)
        assert target_mortgage[0] == 0.0
        assert target_mortgage[1] == 100000.0

    def test__target_consumption_uses_income_override(self, test_households, monkeypatch):
        n_households = test_households.ts.current("n_households")
        n_industries = test_households.n_industries
        income_override = np.full(n_households, 123.0)
        lagged_income = np.full(n_households, 77.0)
        current_income = np.full(n_households, 88.0)
        test_households.ts.expected_income.append(lagged_income)
        test_households.ts.expected_income.append(current_income)
        captured = {}

        def compute_target_consumption(**kwargs):
            captured.update(kwargs)
            return np.zeros((n_households, n_industries))

        monkeypatch.setattr(
            test_households.functions["consumption"],
            "compute_target_consumption",
            compute_target_consumption,
        )

        test_households.compute_target_consumption(
            expected_inflation=0.0,
            current_cpi=1.0,
            initial_cpi=1.0,
            exogenous_total_consumption=np.array([0.0]),
            per_capita_unemployment_benefits=0.0,
            tau_vat=0.0,
            assume_zero_growth=False,
            income_override=income_override,
        )

        np.testing.assert_allclose(captured["income"], income_override)
        np.testing.assert_allclose(captured["lagged_income"], lagged_income)

    def test__target_consumption_does_not_compute_learning_for_default_rule(self, test_households, monkeypatch):
        n_households = test_households.ts.current("n_households")
        n_industries = test_households.n_industries
        test_households.states["income_belief_priors"] = {
            "income_belief_mu": np.ones(n_households),
            "income_belief_p": np.ones(n_households),
            "income_belief_rho": np.ones(n_households),
            "sigma2_xi": np.ones(n_households),
            "sigma2_v": np.ones(n_households),
        }
        captured = {}

        def compute_target_consumption(**kwargs):
            captured.update(kwargs)
            return np.zeros((n_households, n_industries))

        monkeypatch.setattr(
            test_households.functions["consumption"],
            "compute_target_consumption",
            compute_target_consumption,
        )

        test_households.compute_target_consumption(
            expected_inflation=0.0,
            current_cpi=1.0,
            initial_cpi=1.0,
            exogenous_total_consumption=np.array([0.0]),
            per_capita_unemployment_benefits=0.0,
            tau_vat=0.0,
            assume_zero_growth=False,
        )

        assert captured["permanent_income_log_ratio"] is None
        assert captured["uncertainty_delta"] is None
        assert "income_belief_runtime_state" not in test_households.states

    def test__update_income_belief_learning_state_persists_diagnostic_flags(self, test_households):
        n_households = test_households.ts.current("n_households")
        test_households.states["income_belief_priors"] = {
            "income_belief_mu": np.zeros(n_households),
            "income_belief_p": np.zeros(n_households),
            "income_belief_rho": np.full(n_households, 0.5),
            "sigma2_v": np.ones(n_households),
            "sigma2_xi": np.full(n_households, 3.0),
        }

        # No lagged_income yet (warm-up period): diagnostics should append zeros,
        # not raise or leave the series un-appended.
        before = len(test_households.ts.historic("income_belief_growth_clipped"))
        test_households.update_income_belief_learning_state(
            current_income=np.full(n_households, 100.0),
            lagged_income=None,
        )
        assert len(test_households.ts.historic("income_belief_growth_clipped")) == before + 1
        np.testing.assert_allclose(test_households.ts.current("income_belief_growth_clipped"), 0.0)
        np.testing.assert_allclose(test_households.ts.current("income_belief_floor_used"), 0.0)
        np.testing.assert_allclose(test_households.ts.current("income_belief_posterior_fallback_used"), 0.0)

        # One household has a near-zero current income (e.g. a bad financial-asset
        # draw under the old, now-fixed signal source) -- its growth should be both
        # floored and clipped; the other household's normal growth should be neither.
        current_income = np.array([1e-13, 110.0] + [100.0] * (n_households - 2))
        lagged_income = np.full(n_households, 100.0)
        test_households.update_income_belief_learning_state(
            current_income=current_income,
            lagged_income=lagged_income,
        )
        assert test_households.ts.current("income_belief_floor_used")[0] == 1.0
        assert test_households.ts.current("income_belief_growth_clipped")[0] == 1.0
        assert test_households.ts.current("income_belief_floor_used")[1] == 0.0
        assert test_households.ts.current("income_belief_growth_clipped")[1] == 0.0

    def test__target_consumption_does_not_auto_wire_runtime_learning_state_for_opt_in_rule(self, test_households):
        n_households = test_households.ts.current("n_households")
        n_industries = test_households.n_industries
        lagged_income = np.linspace(100.0, 120.0, n_households)
        current_income = lagged_income * np.linspace(1.05, 1.15, n_households)
        test_households.ts.expected_income.append(lagged_income)
        test_households.ts.expected_income.append(current_income)
        test_households.states["income_belief_priors"] = {
            "income_belief_mu": np.zeros(n_households),
            "income_belief_p": np.zeros(n_households),
            "income_belief_rho": np.full(n_households, 0.5),
            "sigma2_v": np.ones(n_households),
            "sigma2_xi": np.full(n_households, 3.0),
        }
        captured = {}

        class OptInConsumption:
            uses_income_belief_learning = True
            last_target_consumption_components = None
            last_formula_implied_mpc = None

            def compute_target_consumption(self, **kwargs):
                captured.update(kwargs)
                return np.zeros((n_households, n_industries))

        test_households.functions["consumption"] = OptInConsumption()
        test_households.update_income_belief_learning_state(
            current_income=current_income,
            lagged_income=lagged_income,
        )

        test_households.compute_target_consumption(
            expected_inflation=0.0,
            current_cpi=1.0,
            initial_cpi=1.0,
            exogenous_total_consumption=np.array([0.0]),
            per_capita_unemployment_benefits=0.0,
            tau_vat=0.0,
            assume_zero_growth=False,
        )

        expected_growth = np.log(current_income) - np.log(lagged_income)
        expected_signal = expected_growth - expected_growth.mean()
        expected_kalman_gain = 1.0 / 4.0
        expected_posterior_mean = expected_kalman_gain * expected_signal
        expected_posterior_variance = 0.75
        assert captured["permanent_income_log_ratio"] is None
        assert captured["uncertainty_delta"] is None
        runtime_state = test_households.states["income_belief_runtime_state"]
        assert set(runtime_state) == {"posterior_mean", "posterior_variance"}
        np.testing.assert_allclose(runtime_state["posterior_mean"], expected_posterior_mean)
        np.testing.assert_allclose(runtime_state["posterior_variance"], expected_posterior_variance)
        state_after_planning = {key: value.copy() for key, value in runtime_state.items()}

        next_income = current_income * np.linspace(1.02, 1.08, n_households)
        test_households.ts.expected_income.append(next_income)
        test_households.compute_target_consumption(
            expected_inflation=0.0,
            current_cpi=1.0,
            initial_cpi=1.0,
            exogenous_total_consumption=np.array([0.0]),
            per_capita_unemployment_benefits=0.0,
            tau_vat=0.0,
            assume_zero_growth=False,
        )

        for key, value in state_after_planning.items():
            np.testing.assert_allclose(runtime_state[key], value)

        test_households.update_income_belief_learning_state(
            current_income=next_income,
            lagged_income=current_income,
        )
        test_households.compute_target_consumption(
            expected_inflation=0.0,
            current_cpi=1.0,
            initial_cpi=1.0,
            exogenous_total_consumption=np.array([0.0]),
            per_capita_unemployment_benefits=0.0,
            tau_vat=0.0,
            assume_zero_growth=False,
        )

        next_growth = np.log(next_income) - np.log(current_income)
        next_signal = next_growth - next_growth.mean()
        next_predicted_mean = 0.5 * expected_posterior_mean
        next_predicted_variance = 0.25 * expected_posterior_variance + 1.0
        next_kalman_gain = next_predicted_variance / (next_predicted_variance + 3.0)
        next_posterior_mean = next_predicted_mean + next_kalman_gain * (next_signal - next_predicted_mean)
        next_posterior_variance = (1.0 - next_kalman_gain) * next_predicted_variance
        assert captured["permanent_income_log_ratio"] is None
        assert captured["uncertainty_delta"] is None
        np.testing.assert_allclose(runtime_state["posterior_mean"], next_posterior_mean)
        np.testing.assert_allclose(runtime_state["posterior_variance"], next_posterior_variance)

    def test__credit_augmented_consumption_keeps_stage_3_terms_default_for_increment_2(self, test_households):
        n_households = test_households.ts.current("n_households")
        lagged_income = np.linspace(100.0, 120.0, n_households)
        current_income = lagged_income * np.linspace(1.05, 1.15, n_households)
        test_households.ts.expected_income.append(lagged_income)
        test_households.ts.expected_income.append(current_income)
        test_households.states["income_belief_priors"] = {
            "income_belief_mu": np.zeros(n_households),
            "income_belief_p": np.zeros(n_households),
            "income_belief_rho": np.full(n_households, 0.5),
            "sigma2_v": np.ones(n_households),
            "sigma2_xi": np.full(n_households, 3.0),
        }
        test_households.functions["consumption"] = CreditAugmentedConsumption(
            consumption_smoothing_fraction=0.0,
            consumption_smoothing_window=1,
            minimum_consumption_fraction=0.0,
            uses_income_belief_learning=True,
        )
        test_households.update_income_belief_learning_state(
            current_income=current_income,
            lagged_income=lagged_income,
        )

        test_households.compute_target_consumption(
            expected_inflation=0.0,
            current_cpi=1.0,
            initial_cpi=1.0,
            exogenous_total_consumption=np.array([0.0]),
            per_capita_unemployment_benefits=0.0,
            tau_vat=0.0,
            assume_zero_growth=False,
        )

        components = test_households.functions["consumption"].last_target_consumption_components
        np.testing.assert_allclose(components["target_consumption_permanent_income"], 0.0)
        np.testing.assert_allclose(components["target_consumption_uncertainty_delta"], 0.0)
        assert "income_belief_runtime_state" in test_households.states

    def test__current_income_belief_learning_inputs_returns_nonzero_finite_terms(self, test_households):
        # Increment 5: with a configured horizon (delta/S), a non-trivial
        # posterior state, and non-zero common terms, the inputs are finite,
        # non-zero, and match the resolved zeta-based formula. Feeding them
        # through compute_target_consumption populates the Stage 3 diagnostics.
        n_households = test_households.ts.current("n_households")
        lagged_income = np.linspace(100.0, 120.0, n_households)
        current_income = lagged_income * np.linspace(1.05, 1.15, n_households)
        test_households.ts.expected_income.append(lagged_income)
        test_households.ts.expected_income.append(current_income)
        priors = {
            "income_belief_mu": np.zeros(n_households),
            "income_belief_p": np.zeros(n_households),
            "income_belief_rho": np.full(n_households, 0.5),
            "sigma2_v": np.ones(n_households),
            "sigma2_xi": np.full(n_households, 3.0),
        }
        test_households.states["income_belief_priors"] = priors
        test_households.functions["consumption"] = CreditAugmentedConsumption(
            consumption_smoothing_fraction=0.0,
            consumption_smoothing_window=1,
            minimum_consumption_fraction=0.0,
            permanent_income_propensity=0.1,
            uncertainty_propensity=0.1,
            uses_income_belief_learning=True,
            income_belief_learning_horizon={"delta": 0.95, "S": 40},
        )
        # Advance posterior beliefs so income_belief_mu/p are non-trivial.
        test_households.update_income_belief_learning_state(
            current_income=current_income,
            lagged_income=lagged_income,
        )
        runtime_state = test_households.states["income_belief_runtime_state"]
        # Ensure the posterior variance is non-zero so the uncertainty term is non-zero.
        assert np.any(runtime_state["posterior_variance"] != 0.0)

        common_log_ratio = 0.02
        common_forecast_variance = 0.03
        inputs = test_households.current_income_belief_learning_inputs(
            common_permanent_income_log_ratio=common_log_ratio,
            common_forecast_variance=common_forecast_variance,
        )

        zeta = test_households.states["income_belief_zeta"]
        assert zeta == compute_zeta(0.5, 0.95, 40)
        expected_log_ratio = zeta * runtime_state["posterior_mean"] + common_log_ratio
        expected_uncertainty = (zeta**2) * runtime_state["posterior_variance"] + common_forecast_variance
        np.testing.assert_allclose(inputs["permanent_income_log_ratio"], expected_log_ratio)
        np.testing.assert_allclose(
            inputs["permanent_income_log_ratio_individual"], zeta * runtime_state["posterior_mean"]
        )
        np.testing.assert_allclose(
            inputs["permanent_income_log_ratio_common"],
            np.full(n_households, common_log_ratio),
        )
        np.testing.assert_allclose(inputs["uncertainty_delta"], expected_uncertainty)
        assert np.all(np.isfinite(inputs["permanent_income_log_ratio"]))
        assert np.all(np.isfinite(inputs["uncertainty_delta"]))
        assert np.any(inputs["permanent_income_log_ratio"] != 0.0)
        assert np.any(inputs["uncertainty_delta"] != 0.0)

        test_households.compute_target_consumption(
            expected_inflation=0.0,
            current_cpi=1.0,
            initial_cpi=1.0,
            exogenous_total_consumption=np.array([0.0]),
            per_capita_unemployment_benefits=0.0,
            tau_vat=0.0,
            assume_zero_growth=False,
            permanent_income_log_ratio=inputs["permanent_income_log_ratio"],
            permanent_income_log_ratio_individual=inputs["permanent_income_log_ratio_individual"],
            permanent_income_log_ratio_common=inputs["permanent_income_log_ratio_common"],
            uncertainty_delta=inputs["uncertainty_delta"],
        )
        components = test_households.functions["consumption"].last_target_consumption_components
        np.testing.assert_allclose(
            components["target_consumption_permanent_income_log_ratio"],
            expected_log_ratio,
        )
        np.testing.assert_allclose(
            components["target_consumption_permanent_income_log_ratio_individual"],
            zeta * runtime_state["posterior_mean"],
        )
        np.testing.assert_allclose(
            components["target_consumption_permanent_income_log_ratio_common"],
            np.full(n_households, common_log_ratio),
        )
        np.testing.assert_allclose(
            components["target_consumption_uncertainty_delta"],
            expected_uncertainty,
        )
        assert np.any(components["target_consumption_permanent_income"] != 0.0)
        assert np.any(components["target_consumption_uncertainty"] != 0.0)

    def test__current_income_belief_learning_inputs_zero_common_reduces_to_individual(self, test_households):
        # None common terms (forecast unavailable) reduce each output to its pure
        # individual component, matching the pre-Increment-5 fallback contract.
        n_households = test_households.ts.current("n_households")
        lagged_income = np.linspace(100.0, 120.0, n_households)
        current_income = lagged_income * np.linspace(1.05, 1.15, n_households)
        priors = {
            "income_belief_mu": np.linspace(0.05, 0.15, n_households),
            "income_belief_p": np.linspace(0.2, 0.4, n_households),
            "income_belief_rho": np.full(n_households, 0.5),
            "sigma2_v": np.ones(n_households),
            "sigma2_xi": np.full(n_households, 3.0),
        }
        test_households.states["income_belief_priors"] = priors
        test_households.functions["consumption"] = CreditAugmentedConsumption(
            consumption_smoothing_fraction=0.0,
            consumption_smoothing_window=1,
            minimum_consumption_fraction=0.0,
            uses_income_belief_learning=True,
            income_belief_learning_horizon={"delta": 0.95, "S": 40},
        )
        test_households.update_income_belief_learning_state(
            current_income=current_income,
            lagged_income=lagged_income,
        )
        runtime_state = test_households.states["income_belief_runtime_state"]

        inputs = test_households.current_income_belief_learning_inputs()
        zeta = test_households.states["income_belief_zeta"]
        np.testing.assert_allclose(inputs["permanent_income_log_ratio"], zeta * runtime_state["posterior_mean"])
        np.testing.assert_allclose(inputs["uncertainty_delta"], (zeta**2) * runtime_state["posterior_variance"])

    def test__current_income_belief_learning_inputs_raises_when_horizon_unset(self, test_households):
        # No silent default: zeta has economic meaning, so a missing delta/S
        # while income-belief learning is enabled must raise clearly.
        n_households = test_households.ts.current("n_households")
        test_households.states["income_belief_priors"] = {
            "income_belief_mu": np.zeros(n_households),
            "income_belief_p": np.zeros(n_households),
            "income_belief_rho": np.full(n_households, 0.5),
            "sigma2_v": np.ones(n_households),
            "sigma2_xi": np.full(n_households, 3.0),
        }
        test_households.functions["consumption"] = CreditAugmentedConsumption(
            consumption_smoothing_fraction=0.0,
            consumption_smoothing_window=1,
            minimum_consumption_fraction=0.0,
            uses_income_belief_learning=True,
            # income_belief_learning_horizon deliberately omitted.
        )
        with pytest.raises(ValueError, match="income_belief_learning_horizon"):
            test_households.current_income_belief_learning_inputs()

    def test__legacy_nonresolver_prepare_goods_market_clearing_reports_subsistence_shortfall_without_altering_demand(
        self, test_households
    ):
        n_households = test_households.ts.current("n_households")
        n_industries = test_households.n_industries
        target_consumption = np.full((n_households, n_industries), 1.0)
        target_investment = np.zeros((n_households, n_industries))
        current_consumption_budget = target_consumption.sum(axis=1)
        floor = current_consumption_budget + 10.0

        test_households.ts.override_current("target_consumption", target_consumption.copy())
        test_households.ts.override_current("target_investment", target_investment)
        test_households.exchange_rate_usd_to_lcu = 1.0
        test_households.configure_feasibility_resolver(False)

        shortfall = test_households.prepare_goods_market_clearing(
            exchange_rate_usd_to_lcu=1.0,
            subsistence_consumption=floor,
        )

        goods_to_buy = test_households.transactor_buyer_states["Initial Goods"]
        expected_goods_budget = target_consumption + target_investment
        np.testing.assert_allclose(test_households.ts.current("target_consumption"), target_consumption)
        np.testing.assert_allclose(goods_to_buy.sum(axis=1), expected_goods_budget.sum(axis=1))
        np.testing.assert_allclose(shortfall, floor - current_consumption_budget)

    def test__legacy_nonresolver_prepare_goods_market_clearing_ignores_floor_enforcement(
        self,
        test_households,
    ):
        n_households = test_households.ts.current("n_households")
        n_industries = test_households.n_industries
        target_consumption = np.full((n_households, n_industries), 10.0)
        target_investment = np.full((n_households, n_industries), 2.0)
        floor = np.full(n_households, 1.0)
        test_households.configure_feasibility_resolver(False)
        test_households.post_grant_feasible_plan = households_module.PostGrantFeasiblePlan(
            credit_granted=np.full(n_households, 4.0),
            credit_rationing_gap=np.full(n_households, 2.0),
            planned_liquidation_total=np.full(n_households, 3.0),
            residual_shortfall_after_granted_credit=np.full(n_households, 999.0),
        )
        test_households.ts.override_current("target_consumption", target_consumption.copy())
        test_households.ts.override_current("target_investment", target_investment.copy())
        diagnostic_lengths = {
            key: len(test_households.ts.historic(key))
            for key in [
                "consumption_before_floor",
                "residual_shortfall_before_floor",
                "consumption_after_floor",
                "consumption_cut_amount",
                "remaining_subsistence_shortfall",
                "floor_binding",
                "consumer_payment_suspension_needed",
                "consumer_payment_suspension_amount",
                "mortgage_payment_suspension_needed",
                "mortgage_payment_suspension_amount",
            ]
        }

        shortfall = test_households.prepare_goods_market_clearing(
            exchange_rate_usd_to_lcu=1.0,
            subsistence_consumption=floor,
        )

        goods_to_buy = test_households.transactor_buyer_states["Initial Goods"]
        np.testing.assert_allclose(goods_to_buy, target_consumption + target_investment)
        np.testing.assert_allclose(shortfall, np.zeros(n_households))
        assert test_households.post_grant_feasible_plan.consumption_after_floor is None
        assert diagnostic_lengths == {key: len(test_households.ts.historic(key)) for key in diagnostic_lengths}

    def test__prepare_goods_market_clearing_applies_floor_by_proportional_consumption_scaling(
        self,
        test_households,
    ):
        n_households = test_households.ts.current("n_households")
        n_industries = test_households.n_industries
        basket_weights = np.arange(n_industries, 0, -1, dtype=float)
        target_consumption = np.tile(100.0 * basket_weights / basket_weights.sum(), (n_households, 1))
        target_investment = np.full((n_households, n_industries), 5.0)
        residual_shortfall = np.full(n_households, 30.0)
        floor = np.full(n_households, 80.0)
        test_households.configure_feasibility_resolver(True)
        test_households.post_grant_feasible_plan = households_module.PostGrantFeasiblePlan(
            credit_granted=np.full(n_households, 4.0),
            credit_rationing_gap=np.full(n_households, 2.0),
            planned_liquidation_total=np.full(n_households, 3.0),
            residual_shortfall_after_granted_credit=residual_shortfall,
        )
        test_households.ts.override_current("target_consumption", target_consumption.copy())
        test_households.ts.override_current("target_investment", target_investment.copy())
        test_households.exchange_rate_usd_to_lcu = 1.0

        shortfall = test_households.prepare_goods_market_clearing(
            exchange_rate_usd_to_lcu=1.0,
            subsistence_consumption=floor,
        )

        plan = test_households.post_grant_feasible_plan
        goods_to_buy = test_households.transactor_buyer_states["Initial Goods"]
        consumption_to_buy = goods_to_buy - target_investment
        np.testing.assert_allclose(plan.consumption_before_floor, target_consumption.sum(axis=1))
        np.testing.assert_allclose(plan.consumption_cut_amount, np.full(n_households, 20.0))
        np.testing.assert_allclose(plan.consumption_after_floor, np.full(n_households, 80.0))
        np.testing.assert_allclose(plan.remaining_subsistence_shortfall, np.full(n_households, 10.0))
        np.testing.assert_allclose(consumption_to_buy.sum(axis=1), plan.consumption_after_floor)
        np.testing.assert_allclose(
            consumption_to_buy,
            target_consumption * (plan.consumption_after_floor / target_consumption.sum(axis=1))[:, None],
        )
        np.testing.assert_allclose(test_households.ts.current("target_consumption"), consumption_to_buy)
        np.testing.assert_allclose(goods_to_buy - consumption_to_buy, target_investment)
        np.testing.assert_allclose(shortfall, plan.remaining_subsistence_shortfall)

        test_households.ts.override_current("nominal_amount_spent_in_lcu", goods_to_buy.copy())
        test_households.update_consumption_and_investment(tau_vat=0.0, tau_cf=0.0)
        np.testing.assert_allclose(test_households.ts.current("consumption"), plan.consumption_after_floor)
        np.testing.assert_allclose(test_households.ts.current("investment"), target_investment)

    def test__prepare_goods_market_clearing_persists_consumption_cut_reconciliation(
        self,
        test_households,
    ):
        n_households = test_households.ts.current("n_households")
        n_industries = test_households.n_industries
        target_consumption = np.full((n_households, n_industries), 25.0)
        target_investment = np.full((n_households, n_industries), 4.0)
        residual_shortfall = np.resize(np.asarray([10.0, 35.0, 0.0, 12.0]), n_households)
        floor = np.full(n_households, 80.0)
        test_households.configure_feasibility_resolver(True)
        test_households.post_grant_feasible_plan = households_module.PostGrantFeasiblePlan(
            credit_granted=np.full(n_households, 4.0),
            credit_rationing_gap=np.full(n_households, 2.0),
            planned_liquidation_total=np.full(n_households, 3.0),
            residual_shortfall_after_granted_credit=residual_shortfall,
        )
        test_households.ts.override_current("target_consumption", target_consumption.copy())
        test_households.ts.override_current("target_investment", target_investment.copy())
        test_households.exchange_rate_usd_to_lcu = 1.0

        test_households.prepare_goods_market_clearing(
            exchange_rate_usd_to_lcu=1.0,
            subsistence_consumption=floor,
        )

        persisted_before = test_households.ts.current("consumption_before_floor")
        persisted_after = test_households.ts.current("consumption_after_floor")
        persisted_cut = test_households.ts.current("consumption_cut_amount")

        np.testing.assert_allclose(persisted_before, target_consumption.sum(axis=1))
        np.testing.assert_allclose(
            persisted_cut,
            persisted_before - persisted_after,
        )
        np.testing.assert_allclose(
            persisted_cut,
            test_households.post_grant_feasible_plan.consumption_cut_amount,
        )

    def test__prepare_goods_market_clearing_floor_does_not_mutate_credit_liquidation_or_balance_sheet(
        self,
        test_households,
    ):
        n_households = test_households.ts.current("n_households")
        n_industries = test_households.n_industries
        target_consumption = np.full((n_households, n_industries), 20.0)
        target_investment = np.full((n_households, n_industries), 3.0)
        test_households.configure_feasibility_resolver(True)
        test_households.post_grant_feasible_plan = households_module.PostGrantFeasiblePlan(
            credit_granted=np.full(n_households, 4.0),
            credit_rationing_gap=np.full(n_households, 2.0),
            planned_liquidation_total=np.full(n_households, 3.0),
            residual_shortfall_after_granted_credit=np.full(n_households, 30.0),
        )
        test_households.ts.override_current("target_consumption", target_consumption.copy())
        test_households.ts.override_current("target_investment", target_investment.copy())
        invariant_keys = [
            "target_consumption_loans",
            "received_consumption_loans",
            "liquidation_planned",
            "liquid_financial_assets",
            "wealth_financial_assets",
            "wealth_real_assets",
            "wealth",
            "consumption_loan_debt",
            "mortgage_debt",
            "debt",
        ]
        before = {key: np.asarray(test_households.ts.current(key)).copy() for key in invariant_keys}
        plan_before = test_households.post_grant_feasible_plan
        credit_granted_before = plan_before.credit_granted.copy()
        credit_rationing_gap_before = plan_before.credit_rationing_gap.copy()
        planned_liquidation_before = plan_before.planned_liquidation_total.copy()
        residual_shortfall_before = plan_before.residual_shortfall_after_granted_credit.copy()

        test_households.prepare_goods_market_clearing(
            exchange_rate_usd_to_lcu=1.0,
            subsistence_consumption=np.full(n_households, 50.0),
        )

        plan_after = test_households.post_grant_feasible_plan
        for key, value in before.items():
            np.testing.assert_allclose(test_households.ts.current(key), value, equal_nan=True)
        np.testing.assert_allclose(plan_after.credit_granted, credit_granted_before)
        np.testing.assert_allclose(plan_after.credit_rationing_gap, credit_rationing_gap_before)
        np.testing.assert_allclose(plan_after.planned_liquidation_total, planned_liquidation_before)
        np.testing.assert_allclose(plan_after.residual_shortfall_after_granted_credit, residual_shortfall_before)

    def test__prepare_goods_market_clearing_requires_post_grant_plan_when_resolver_enabled(
        self,
        test_households,
    ):
        n_households = test_households.ts.current("n_households")
        n_industries = test_households.n_industries
        test_households.configure_feasibility_resolver(True)
        test_households.ts.override_current("target_consumption", np.ones((n_households, n_industries)))
        test_households.ts.override_current("target_investment", np.zeros((n_households, n_industries)))

        with pytest.raises(RuntimeError, match="post_grant_feasible_plan"):
            test_households.prepare_goods_market_clearing(
                exchange_rate_usd_to_lcu=1.0,
                subsistence_consumption=np.zeros(n_households),
            )

    def test__prepare_goods_market_clearing_requires_subsistence_floor_when_resolver_enabled(
        self,
        test_households,
    ):
        n_households = test_households.ts.current("n_households")
        n_industries = test_households.n_industries
        test_households.configure_feasibility_resolver(True)
        test_households.post_grant_feasible_plan = households_module.PostGrantFeasiblePlan(
            credit_granted=np.zeros(n_households),
            credit_rationing_gap=np.zeros(n_households),
            planned_liquidation_total=np.zeros(n_households),
            residual_shortfall_after_granted_credit=np.zeros(n_households),
        )
        test_households.ts.override_current("target_consumption", np.ones((n_households, n_industries)))
        test_households.ts.override_current("target_investment", np.zeros((n_households, n_industries)))

        with pytest.raises(RuntimeError, match="subsistence_consumption"):
            test_households.prepare_goods_market_clearing(
                exchange_rate_usd_to_lcu=1.0,
                subsistence_consumption=None,
            )

    def test__paper_asset_returns_do_not_override_expected_financial_income(self, test_households, monkeypatch):
        from macromodel.agents.households.func.wealth import PaperAssetReturnWealthSetter

        n_households = test_households.ts.current("n_households")
        expected_financial_income = np.full(n_households, 7.0)
        test_households.functions["wealth"] = PaperAssetReturnWealthSetter(
            other_real_assets_depreciation_rate=0.05,
            mu_eq=0.10,
            mu_bond=0.02,
            sigma_eq=0.20,
            sigma_bond=0.10,
            rho=0.25,
            equity_weight=0.75,
        )

        monkeypatch.setattr(
            test_households.functions["financial_assets"],
            "compute_expected_income",
            lambda **kwargs: expected_financial_income,
        )

        np.testing.assert_allclose(
            test_households.compute_expected_income_from_financial_assets(),
            expected_financial_income,
        )

    def test__paper_asset_returns_are_excluded_from_expected_and_realised_income(self, test_households):
        from macromodel.agents.households.func.wealth import PaperAssetReturnWealthSetter

        n_households = test_households.ts.current("n_households")
        test_households.functions["wealth"] = PaperAssetReturnWealthSetter(
            other_real_assets_depreciation_rate=0.05,
            mu_eq=0.10,
            mu_bond=0.02,
            sigma_eq=0.20,
            sigma_bond=0.10,
            rho=0.25,
            equity_weight=0.75,
        )
        components = {
            "expected_income_employee": 2.0,
            "expected_income_social_transfers": 3.0,
            "income_employee": 5.0,
            "income_social_transfers": 7.0,
            "income_rental": 11.0,
        }
        for field, value in components.items():
            test_households.ts.override_current(field, np.full(n_households, value))
        test_households.ts.override_current("expected_income_financial_assets", np.full(n_households, 13.0))
        test_households.ts.override_current("income_financial_assets", np.full(n_households, 17.0))

        np.testing.assert_allclose(test_households.compute_expected_income(), np.full(n_households, 16.0))
        np.testing.assert_allclose(test_households.compute_income(), np.full(n_households, 23.0))

    def test__paper_asset_dividends_enter_expected_and_realised_income(self, test_households):
        from macromodel.agents.households.func.wealth import PaperAssetReturnWealthSetter

        n_households = test_households.ts.current("n_households")
        test_households.functions["wealth"] = PaperAssetReturnWealthSetter(
            other_real_assets_depreciation_rate=0.05,
            mu_eq=0.10,
            mu_bond=0.02,
            sigma_eq=0.20,
            sigma_bond=0.10,
            rho=0.25,
            equity_weight=0.75,
        )
        components = {
            "expected_income_employee": 2.0,
            "expected_income_social_transfers": 3.0,
            "income_employee": 5.0,
            "income_social_transfers": 7.0,
            "income_rental": 11.0,
            "expected_income_dividend_distributions": 13.0,
            "income_dividend_distributions": 17.0,
            "income_financial_assets": 19.0,
        }
        for field, value in components.items():
            test_households.ts.override_current(field, np.full(n_households, value))

        np.testing.assert_allclose(test_households.compute_expected_income(), np.full(n_households, 29.0))
        np.testing.assert_allclose(test_households.compute_income(), np.full(n_households, 40.0))

    # def test__households_ts(self, test_households):
    #     for ts_key in [
    #         "n_households",
    #         "target_consumption_before_ce",
    #         "target_consumption_ce",
    #         "target_consumption",
    #         "amount_bought",
    #         "consumption",
    #         "investment_in_other_real_assets",
    #         "income",
    #         "income_employee",
    #         "income_social_transfers",
    #         "income_rental",
    #         "price_paid_for_property",
    #         "rent",
    #         "max_price_willing_to_pay",
    #         "max_rent_willing_to_pay",
    #         "wealth",
    #         "wealth_real_assets",
    #         "wealth_main_residence",
    #         "wealth_other_properties",
    #         "wealth_other_real_assets",
    #         "liquid_financial_assets",
    #         "illiquid_financial_assets",
    #         "wealth_financial_assets",
    #         "payday_loan_debt",
    #         "consumption_expansion_loan_debt",
    #         "mortgage_debt",
    #         "debt",
    #         "net_wealth",
    #         "target_payday_loans",
    #         "received_payday_loans",
    #         "target_consumption_expansion_loans",
    #         "received_consumption_expansion_loans",
    #         "target_mortgage",
    #         "received_mortgages",
    #         "debt_installments",
    #         "interest_paid_on_deposits",
    #         "interest_paid_on_loans",
    #         "interest_paid",
    #     ]:
    #         assert ts_key in test_households.ts.get_keys()
    #
    # def test__get_saving_rates_by_household(self, test_households):
    #     assert np.allclose(test_households.get_saving_rates_by_household(), np.full(18, 0.2))
    #
    # def test__get_social_transfers_by_household(self, test_households):
    #     assert np.allclose(
    #         test_households.compute_social_transfer_income(
    #             total_other_social_transfers=1000.0,
    #             central_government_init={
    #                 "functions": {
    #                     "household_social_transfers": {
    #                         "parameters": {
    #                             "independents": {"value": []},
    #                             "steps": {"value": 1},
    #                         }
    #                     }
    #                 }
    #             },
    #         ),
    #         np.full(18, 55.55555556),
    #     )


class TestHouseholdsUpdateConsumptionEmissions:
    """Tests for the emission multiplier logic in update_consumption_and_investment."""

    def test__appends_consumption_emissions(self, test_households):
        n_hh = len(test_households.states["Type"])
        n_industries = test_households.n_industries
        _setup_emission_ts(test_households, n_hh, n_industries)
        n_before = len(test_households.ts.consumption_emissions)
        emitting_indices = np.array([0, 1, 2, 3])

        test_households.update_consumption_and_investment(
            tau_vat=0.0,
            tau_cf=0.0,
            add_emissions=True,
            readjusted_factors=np.ones(4),
            emitting_indices=emitting_indices,
        )

        assert len(test_households.ts.consumption_emissions) == n_before + 1
        assert len(test_households.ts.investment_emissions) == n_before + 1

    def test__consumption_emissions_by_good_appended(self, test_households):
        n_hh = len(test_households.states["Type"])
        n_industries = test_households.n_industries
        _setup_emission_ts(test_households, n_hh, n_industries)
        emitting_indices = np.array([0, 1, 2, 3])

        test_households.update_consumption_and_investment(
            tau_vat=0.0,
            tau_cf=0.0,
            add_emissions=True,
            readjusted_factors=np.ones(4),
            emitting_indices=emitting_indices,
        )

        assert len(test_households.ts.consumption_emissions_by_good) == 2
        assert len(test_households.ts.investment_emissions_by_good) == 2

    def test__all_ones_multiplier_matches_no_multiplier(self, test_households):
        """Consumption fractions of 1.0 everywhere should give same emissions as no multiplier."""
        n_hh = len(test_households.states["Type"])
        n_industries = test_households.n_industries
        _setup_emission_ts(test_households, n_hh, n_industries)
        emitting_indices = np.array([0, 1, 2, 3])
        readjusted_factors = np.ones(4)

        test_households.update_consumption_and_investment(
            tau_vat=0.0,
            tau_cf=0.0,
            add_emissions=True,
            readjusted_factors=readjusted_factors,
            emitting_indices=emitting_indices,
            use_emission_multiplier=False,
        )
        baseline = test_households.ts.consumption_emissions[-1].copy()

        consumption_ones = np.ones((1, n_industries))
        investment_ones = np.ones((1, n_industries))
        test_households.emission_fractions = EmissionFractions(consumption=consumption_ones, investment=investment_ones)
        test_households.update_consumption_and_investment(
            tau_vat=0.0,
            tau_cf=0.0,
            add_emissions=True,
            readjusted_factors=readjusted_factors,
            emitting_indices=emitting_indices,
            use_emission_multiplier=True,
        )

        assert np.allclose(baseline, test_households.ts.consumption_emissions[-1])

    def test__zero_multiplier_yields_zero_emissions(self, test_households):
        """Consumption fractions of 0.0 should zero out emissions."""
        n_hh = len(test_households.states["Type"])
        n_industries = test_households.n_industries
        _setup_emission_ts(test_households, n_hh, n_industries)
        emitting_indices = np.array([0, 1, 2, 3])

        consumption_zeros = np.zeros((1, n_industries))
        investment_zeros = np.zeros((1, n_industries))
        test_households.emission_fractions = EmissionFractions(
            consumption=consumption_zeros, investment=investment_zeros
        )
        test_households.update_consumption_and_investment(
            tau_vat=0.0,
            tau_cf=0.0,
            add_emissions=True,
            readjusted_factors=np.ones(4),
            emitting_indices=emitting_indices,
            use_emission_multiplier=True,
        )

        assert np.allclose(test_households.ts.consumption_emissions[-1], 0.0)
        assert np.allclose(test_households.ts.investment_emissions[-1], 0.0)

    def test__ch4_by_good_populated_when_ch4_factors_provided(self, test_households):
        n_hh = len(test_households.states["Type"])
        n_industries = test_households.n_industries
        _setup_emission_ts(test_households, n_hh, n_industries)
        test_households.ts.dicts["consumption_emissions_ch4_by_good"] = [np.zeros(n_industries)]
        test_households.ts.dicts["investment_emissions_ch4_by_good"] = [np.zeros(n_industries)]
        emitting_indices = np.array([0, 1, 2, 3])

        test_households.update_consumption_and_investment(
            tau_vat=0.0,
            tau_cf=0.0,
            add_emissions=True,
            readjusted_factors=np.ones(4),
            emitting_indices=emitting_indices,
            readjusted_factors_ch4=np.ones(4),
            emitting_indices_ch4=np.array([0, 1, 2, 3]),
        )

        assert len(test_households.ts.consumption_emissions_ch4_by_good) == 2
        assert len(test_households.ts.investment_emissions_ch4_by_good) == 2


class TestComputeStage4PortfolioDiagnostics:
    """Regression guard: Stage 4 diagnostics are diagnostics-only.

    ``compute_stage4_portfolio_diagnostics`` must never mutate the core
    balance-sheet time series it reads from (``liquid_financial_assets``,
    ``illiquid_financial_assets``, ``wealth_financial_assets``,
    ``wealth``, ``income``, ``consumption``, ``debt_installments``). It is
    only reachable when ``uses_portfolio_choice=True`` on the wealth
    function, so the fixture's setter is swapped for a
    ``PaperAssetReturnWealthSetter`` configured that way.
    """

    def _enable_portfolio_choice(self, test_households):
        test_households.functions["wealth"] = PaperAssetReturnWealthSetter(
            other_real_assets_depreciation_rate=0.05,
            mu_eq=0.0029,
            mu_bond=0.0081,
            sigma_eq=0.0,
            sigma_bond=0.0,
            rho=0.0,
            equity_weight=0.5,
            draw_scope="country_period",
            uses_portfolio_choice=True,
            target_share_source="scalar",
            default_target_illiquid_share=0.65,
            phi_1=5.0,
            lambda_kappa=0.1,
            fixed_cost_share=0.001,
        )

    def test__core_wealth_and_income_series_are_bit_identical_after_call(self, test_households):
        self._enable_portfolio_choice(test_households)
        watched_keys = [
            "liquid_financial_assets",
            "illiquid_financial_assets",
            "wealth_financial_assets",
            "wealth",
            "income",
            "consumption",
            "debt_installments",
        ]
        before = {key: [arr.copy() for arr in test_households.ts.dicts[key]] for key in watched_keys}

        test_households.compute_stage4_portfolio_diagnostics()

        for key in watched_keys:
            after = test_households.ts.dicts[key]
            assert len(after) == len(before[key]), f"{key} series length changed"
            for before_arr, after_arr in zip(before[key], after):
                np.testing.assert_array_equal(before_arr, after_arr, err_msg=f"{key} mutated")

    def test__diagnostic_series_get_exactly_one_new_appended_entry(self, test_households):
        self._enable_portfolio_choice(test_households)
        diagnostic_keys = [
            "portfolio_actual_illiquid_share",
            "portfolio_opening_tfa_scale",
            "portfolio_target_tfa_base",
            "portfolio_post_return_lfa",
            "portfolio_post_return_ifa",
            "portfolio_liquid_return_rate",
            "portfolio_illiquid_return_rate",
            "portfolio_investable_surplus",
            "portfolio_participation_probability",
            "portfolio_participates",
            "portfolio_target_illiquid_share",
            "portfolio_target_illiquid_assets",
            "portfolio_delta_tilde",
            "portfolio_kappa_star_tilde",
            "portfolio_kappa_tilde",
            "portfolio_desired_illiquid_adjustment",
            "portfolio_adjustment_cost",
            "portfolio_counterfactual_lfa_flow",
            "portfolio_counterfactual_ifa_flow",
            "portfolio_inaction_flag",
            "portfolio_upper_bound_flag",
            "portfolio_lower_bound_flag",
            "portfolio_infeasible_interval_flag",
            "portfolio_no_financial_assets_flag",
            "portfolio_target_share_clipped_flag",
            "portfolio_settlement_enabled",
        ]
        before_lengths = {key: len(test_households.ts.dicts[key]) for key in diagnostic_keys}

        test_households.compute_stage4_portfolio_diagnostics()

        for key in diagnostic_keys:
            assert len(test_households.ts.dicts[key]) == before_lengths[key] + 1, (
                f"{key} did not get exactly one new entry"
            )
            n_hh = len(test_households.states["Type"])
            assert test_households.ts.dicts[key][-1].shape[0] == n_hh

    def test__opening_tfa_scale_uses_current_period_fallback_at_t0(self, test_households):
        # TimeSeries.prev() falls back to the current value when only one entry
        # exists yet (t=0), so at the very first call opening_tfa_scale equals
        # this period's post_return_ifa + post_surplus_lfa rather than a genuine
        # prior-period stock. This is a known, documented approximation (not a
        # crash) — this test pins the behavior so a future TimeSeries change
        # doesn't silently alter the first-period diagnostic without notice.
        self._enable_portfolio_choice(test_households)

        test_households.compute_stage4_portfolio_diagnostics()

        expected = test_households.ts.current("illiquid_financial_assets") + test_households.ts.current(
            "liquid_financial_assets"
        )
        np.testing.assert_allclose(test_households.ts.dicts["portfolio_opening_tfa_scale"][-1], expected)

    def test__liquid_return_rate_and_participation_probability_stay_nan(self, test_households):
        # Both fields are explicitly not sourced in this increment (see the
        # comments at the two append() call sites in households.py); pin them
        # as NaN so a careless future refactor can't silently start emitting
        # 0.0 (which would look like a real measured rate/probability) instead.
        self._enable_portfolio_choice(test_households)

        test_households.compute_stage4_portfolio_diagnostics()

        assert np.all(np.isnan(test_households.ts.dicts["portfolio_liquid_return_rate"][-1]))
        assert np.all(np.isnan(test_households.ts.dicts["portfolio_participation_probability"][-1]))


class TestHouseholdsUpdateWealthPortfolioSettlement:
    @staticmethod
    def _zero_rebalancing(n_households):
        zeros = np.zeros(n_households)
        return PortfolioRebalancingResult(
            portfolio_participates=np.ones(n_households, dtype=bool),
            actual_illiquid_share=zeros.copy(),
            target_illiquid_assets=zeros.copy(),
            delta_tilde=zeros.copy(),
            kappa_star_tilde=zeros.copy(),
            kappa_tilde=zeros.copy(),
            desired_illiquid_adjustment=zeros.copy(),
            adjustment_cost=zeros.copy(),
            counterfactual_lfa_flow=zeros.copy(),
            counterfactual_ifa_flow=zeros.copy(),
            inaction_flag=np.ones(n_households, dtype=bool),
            upper_bound_flag=np.zeros(n_households, dtype=bool),
            lower_bound_flag=np.zeros(n_households, dtype=bool),
            infeasible_interval_flag=np.zeros(n_households, dtype=bool),
            no_financial_assets_flag=np.zeros(n_households, dtype=bool),
            portfolio_valid_flag=np.ones(n_households, dtype=bool),
        )

    def _configure_update_wealth(
        self,
        test_households,
        monkeypatch,
        *,
        resolver,
        settles,
        use_actual_diagnostics=False,
        use_real_liquidation=False,
    ):
        n_households = len(test_households.states["Type"])
        zeros = np.zeros(n_households)
        settlement_calls = []

        class WealthStub:
            exclude_financial_asset_income_from_saving = False
            uses_periodic_illiquid_returns = False
            uses_portfolio_choice = True
            settles_portfolio_choice = settles
            target_share_source = "scalar"
            default_target_illiquid_share = 0.5
            phi_1 = 5.0
            lambda_kappa = 0.1
            fixed_cost_share = 0.0

            @staticmethod
            def distribute_new_wealth(**_kwargs):
                return zeros.copy(), zeros.copy()

            @staticmethod
            def use_up_wealth(**_kwargs):
                return zeros.copy(), zeros.copy()

        monkeypatch.setitem(test_households.functions, "wealth", WealthStub())
        test_households.uses_feasibility_resolver = resolver
        test_households.ts.override_current("income", np.full(n_households, 100.0))
        test_households.ts.override_current("expected_income", np.full(n_households, 100.0))
        test_households.ts.override_current("income_financial_assets", zeros.copy())
        test_households.ts.override_current("rent", zeros.copy())
        test_households.ts.override_current("liquid_financial_assets", zeros.copy())
        test_households.ts.override_current("illiquid_financial_assets", np.full(n_households, 50.0))
        for field in (
            "interest_paid",
            "price_paid_for_property",
            "debt_installments",
            "received_consumption_loans",
            "received_mortgages",
        ):
            test_households.ts.override_current(field, zeros.copy())
        test_households.ts.override_current("investment", np.zeros_like(test_households.ts.current("investment")))
        test_households.ts.override_current(
            "nominal_amount_spent_in_lcu",
            np.zeros_like(test_households.ts.current("nominal_amount_spent_in_lcu")),
        )
        test_households.ts.override_current(
            "target_consumption",
            np.zeros_like(test_households.ts.current("target_consumption")),
        )
        test_households.ts.override_current("household_saving", np.zeros(n_households))

        monkeypatch.setattr(
            test_households,
            "compute_wealth_of_the_main_residence",
            lambda *, housing_data: zeros.copy(),
        )
        monkeypatch.setattr(
            test_households,
            "compute_wealth_of_other_properties",
            lambda *, housing_data: zeros.copy(),
        )
        monkeypatch.setattr(test_households, "compute_wealth_of_other_real_assets", lambda: zeros.copy())
        monkeypatch.setattr(
            test_households,
            "compute_wealth_of_other_financial_assets",
            lambda current_wealth_in_other_financial_assets=None, **_kwargs: (
                np.full(n_households, 50.0)
                if current_wealth_in_other_financial_assets is None
                else current_wealth_in_other_financial_assets
            ),
        )
        monkeypatch.setattr(
            test_households,
            "compute_wealth_in_deposits",
            lambda **_kwargs: np.full(n_households, 100.0),
        )
        monkeypatch.setattr(test_households, "current_illiquid_financial_asset_return_rate", lambda: 0.0)

        if not use_actual_diagnostics:
            rebalancing = self._zero_rebalancing(n_households)
            diagnostics = Stage4HouseholdDiagnostics(
                portfolio_opening_tfa_scale=np.full(n_households, 150.0),
                portfolio_target_tfa_base=np.full(n_households, 150.0),
                portfolio_post_return_lfa=np.full(n_households, 108.0),
                portfolio_post_return_ifa=np.full(n_households, 42.0),
                portfolio_investable_surplus=np.full(n_households, 10.0),
                portfolio_target_illiquid_share=np.full(n_households, 0.5),
                portfolio_target_share_clipped_flag=np.zeros(n_households, dtype=bool),
                rebalancing=rebalancing,
            )
            monkeypatch.setattr(
                test_households,
                "compute_stage4_portfolio_diagnostics",
                lambda **_kwargs: diagnostics,
            )

        if resolver:
            if use_real_liquidation:
                test_households.post_grant_feasible_plan = PostGrantFeasiblePlan(
                    funded_from_liquid_assets=np.zeros(n_households),
                    credit_granted=np.zeros(n_households),
                    credit_rationing_gap=np.zeros(n_households),
                    planned_liquidation_total=np.zeros(n_households),
                    reserved_liquidation_total=np.zeros(n_households),
                    residual_shortfall_after_granted_credit=np.zeros(n_households),
                )
            else:
                test_households.post_grant_feasible_plan = SimpleNamespace(
                    funded_from_liquid_assets=np.zeros(n_households),
                    credit_granted=np.zeros(n_households),
                    post_liquidation_lfa=np.full(n_households, 108.0),
                    post_liquidation_ifa=np.full(n_households, 42.0),
                    settled_liquidation_total=np.full(n_households, 8.0),
                )

            if not use_real_liquidation:

                def expose_post_liquidation_bases(*, base_lfa, base_ifa):
                    settlement_calls.append((base_lfa.copy(), base_ifa.copy()))
                    return np.full(n_households, 108.0), np.full(n_households, 42.0)

                monkeypatch.setattr(test_households, "settle_post_grant_liquidation", expose_post_liquidation_bases)
        else:
            test_households.post_grant_feasible_plan = None

        return n_households, settlement_calls

    def test__settled_update_consumes_stage5_bases_and_persists_once(self, test_households, monkeypatch):
        n_households, settlement_calls = self._configure_update_wealth(
            test_households,
            monkeypatch,
            resolver=True,
            settles=True,
        )
        initial_lfa_length = len(test_households.ts.dicts["liquid_financial_assets"])
        initial_ifa_length = len(test_households.ts.dicts["illiquid_financial_assets"])
        shadow_append_observations = []
        stage4_inputs = []
        append_shadow_diagnostics = test_households._append_stage4_portfolio_diagnostics
        compute_stage4_portfolio_diagnostics = test_households.compute_stage4_portfolio_diagnostics

        def observe_stage4_inputs(**kwargs):
            stage4_inputs.append(kwargs)
            return compute_stage4_portfolio_diagnostics(**kwargs)

        def observe_shadow_append(diagnostics):
            shadow_append_observations.append(
                (
                    len(test_households.ts.dicts["liquid_financial_assets"]),
                    len(test_households.ts.dicts["illiquid_financial_assets"]),
                )
            )
            append_shadow_diagnostics(diagnostics)

        monkeypatch.setattr(test_households, "_append_stage4_portfolio_diagnostics", observe_shadow_append)
        monkeypatch.setattr(test_households, "compute_stage4_portfolio_diagnostics", observe_stage4_inputs)

        test_households.update_wealth(housing_data=pd.DataFrame(), tau_cf=0.0)

        assert len(settlement_calls) == 1
        assert shadow_append_observations == [(initial_lfa_length + 1, initial_ifa_length + 1)]
        assert len(stage4_inputs) == 1
        np.testing.assert_allclose(stage4_inputs[0]["post_surplus_lfa"], np.full(n_households, 108.0))
        np.testing.assert_allclose(stage4_inputs[0]["post_return_ifa"], np.full(n_households, 42.0))
        np.testing.assert_allclose(settlement_calls[0][0], np.full(n_households, 100.0))
        np.testing.assert_allclose(settlement_calls[0][1], np.full(n_households, 50.0))
        assert len(test_households.ts.dicts["liquid_financial_assets"]) == initial_lfa_length + 1
        assert len(test_households.ts.dicts["illiquid_financial_assets"]) == initial_ifa_length + 1
        np.testing.assert_allclose(test_households.ts.current("liquid_financial_assets"), np.full(n_households, 108.0))
        np.testing.assert_allclose(
            test_households.ts.current("illiquid_financial_assets"),
            np.full(n_households, 42.0),
        )
        zero_flows = np.zeros(n_households)
        np.testing.assert_allclose(
            test_households.ts.current("portfolio_settlement_committed_lfa_flow"),
            zero_flows,
        )
        np.testing.assert_allclose(test_households.ts.current("portfolio_settlement_committed_ifa_flow"), zero_flows)
        np.testing.assert_allclose(
            test_households.ts.current("portfolio_settlement_status"), np.full(n_households, 2.0)
        )

    def test__resolver_bypasses_legacy_use_up_wealth(self, test_households, monkeypatch):
        self._configure_update_wealth(test_households, monkeypatch, resolver=True, settles=True)

        def fail_if_called(**_kwargs):
            raise AssertionError("legacy use_up_wealth must not run with the feasibility resolver")

        monkeypatch.setattr(test_households.functions["wealth"], "use_up_wealth", fail_if_called)
        test_households.update_wealth(housing_data=pd.DataFrame(), tau_cf=0.0)

    def test__resolver_settles_qexec_before_applying_same_period_ifa_return(self, test_households, monkeypatch):
        n_households, _ = self._configure_update_wealth(
            test_households,
            monkeypatch,
            resolver=True,
            settles=False,
            use_real_liquidation=True,
        )
        test_households.ts.override_current("income", np.zeros(n_households))
        test_households.ts.override_current("expected_income", np.zeros(n_households))
        test_households.ts.override_current("liquid_financial_assets", np.zeros(n_households))
        test_households.ts.override_current("illiquid_financial_assets", np.full(n_households, 100.0))
        test_households.post_grant_feasible_plan = PostGrantFeasiblePlan(
            funded_from_liquid_assets=np.zeros(n_households),
            credit_granted=np.zeros(n_households),
            credit_rationing_gap=np.zeros(n_households),
            planned_liquidation_total=np.full(n_households, 80.0),
            reserved_liquidation_total=np.full(n_households, 80.0),
            residual_shortfall_after_granted_credit=np.zeros(n_households),
        )
        return_bases = []
        stage4_inputs = []

        def apply_negative_return(*, current_wealth_in_other_financial_assets, **_kwargs):
            return_bases.append(current_wealth_in_other_financial_assets.copy())
            return current_wealth_in_other_financial_assets * 0.7

        diagnostics = test_households.compute_stage4_portfolio_diagnostics()
        monkeypatch.setattr(
            test_households,
            "compute_wealth_of_other_financial_assets",
            apply_negative_return,
        )
        monkeypatch.setattr(
            test_households,
            "compute_stage4_portfolio_diagnostics",
            lambda **kwargs: (stage4_inputs.append(kwargs), diagnostics)[1],
        )

        test_households.update_wealth(housing_data=pd.DataFrame(), tau_cf=0.0)

        np.testing.assert_allclose(return_bases, [np.full(n_households, 20.0)])
        np.testing.assert_allclose(stage4_inputs[0]["post_return_ifa"], np.full(n_households, 14.0))
        np.testing.assert_allclose(test_households.ts.current("liquid_financial_assets"), np.full(n_households, 80.0))
        np.testing.assert_allclose(test_households.ts.current("illiquid_financial_assets"), np.full(n_households, 14.0))
        np.testing.assert_allclose(test_households.ts.current("income"), np.zeros(n_households))
        np.testing.assert_allclose(test_households.ts.current("expected_income"), np.zeros(n_households))
        np.testing.assert_allclose(test_households.post_grant_feasible_plan.credit_granted, np.zeros(n_households))
        np.testing.assert_allclose(
            test_households.post_grant_feasible_plan.reserved_liquidation_total, np.full(n_households, 80.0)
        )
        np.testing.assert_allclose(test_households.ts.current("stage5_cash_ledger_residual"), np.zeros(n_households))

    def test__resolver_books_deposit_interest_once_as_a_direct_liquid_return(self, test_households, monkeypatch):
        n_households, _ = self._configure_update_wealth(
            test_households,
            monkeypatch,
            resolver=True,
            settles=False,
            use_real_liquidation=True,
        )
        income = np.full(n_households, 30.0)
        expected_income = np.full(n_households, 40.0)
        opening_lfa = np.full(n_households, 100.0)
        opening_ifa = np.full(n_households, 50.0)
        deposit_interest_received = np.full(n_households, 10.0)
        test_households.ts.override_current("income", income)
        test_households.ts.override_current("expected_income", expected_income)
        test_households.ts.override_current("liquid_financial_assets", opening_lfa)
        test_households.ts.override_current("illiquid_financial_assets", opening_ifa)
        # Accounting convention: an interest receipt is negative household
        # interest paid and therefore a direct addition to LFA.
        test_households.ts.override_current("interest_paid", -deposit_interest_received)

        test_households.update_wealth(housing_data=pd.DataFrame(), tau_cf=0.0)

        np.testing.assert_allclose(
            test_households.ts.current("liquid_financial_assets"), opening_lfa + income + deposit_interest_received
        )
        np.testing.assert_allclose(test_households.ts.current("illiquid_financial_assets"), opening_ifa)
        np.testing.assert_allclose(test_households.ts.current("income"), income)
        np.testing.assert_allclose(test_households.ts.current("expected_income"), expected_income)
        np.testing.assert_allclose(test_households.post_grant_feasible_plan.credit_granted, np.zeros(n_households))
        np.testing.assert_allclose(
            test_households.post_grant_feasible_plan.reserved_liquidation_total, np.zeros(n_households)
        )
        np.testing.assert_allclose(test_households.ts.current("stage5_cash_ledger_residual"), np.zeros(n_households))

    def test__legacy_compatibility_preserves_use_up_wealth_branch(self, test_households, monkeypatch):
        n_households, _ = self._configure_update_wealth(
            test_households,
            monkeypatch,
            resolver=False,
            settles=False,
        )
        calls = []

        def legacy_use_up_wealth(**kwargs):
            calls.append(kwargs)
            return np.zeros(n_households), np.zeros(n_households)

        monkeypatch.setattr(test_households.functions["wealth"], "use_up_wealth", legacy_use_up_wealth)
        test_households.update_wealth(housing_data=pd.DataFrame(), tau_cf=0.0)

        assert len(calls) == 1

    def test__legacy_compatibility_paper_setter_configuration_warns_of_legacy_withdrawal(self, test_households):
        test_households.functions["wealth"] = PaperAssetReturnWealthSetter(
            other_real_assets_depreciation_rate=0.05,
            mu_eq=0.0029,
            mu_bond=0.0081,
            sigma_eq=0.0,
            sigma_bond=0.0,
            rho=0.0,
            equity_weight=0.5,
        )

        with pytest.warns(DeprecationWarning, match="deprecated legacy use_up_wealth"):
            test_households.configure_feasibility_resolver(False)

    def test__resolver_books_realised_cash_difference_only_to_liquid_assets(self, test_households, monkeypatch):
        n_households, _ = self._configure_update_wealth(
            test_households,
            monkeypatch,
            resolver=True,
            settles=False,
        )
        test_households.ts.override_current("expected_income", np.full(n_households, 90.0))
        test_households.ts.override_current("income", np.full(n_households, 100.0))
        target = np.zeros_like(test_households.ts.current("target_consumption"))
        target[:, 0] = 20.0
        realised = np.zeros_like(test_households.ts.current("nominal_amount_spent_in_lcu"))
        realised[:, 0] = 15.0
        test_households.ts.override_current("target_consumption", target)
        test_households.ts.override_current("nominal_amount_spent_in_lcu", realised)
        _replace_post_grant_plan(
            test_households,
            reserved_liquidation_total=np.zeros(n_households),
            settled_liquidation_total=np.zeros(n_households),
        )
        monkeypatch.setattr(
            test_households,
            "settle_post_grant_liquidation",
            lambda *, base_lfa, base_ifa: (base_lfa, base_ifa),
        )

        test_households.update_wealth(housing_data=pd.DataFrame(), tau_cf=0.0)

        np.testing.assert_allclose(test_households.ts.current("realised_cash_flow_adjustment"), 0.0)
        np.testing.assert_allclose(test_households.ts.current("liquid_financial_assets"), 85.0)
        np.testing.assert_allclose(test_households.ts.current("illiquid_financial_assets"), 50.0)

    def test__resolver_books_early_committed_credit_once_in_the_cash_ledger(
        self,
        test_households,
        monkeypatch,
    ):
        n_households, _ = self._configure_update_wealth(
            test_households,
            monkeypatch,
            resolver=True,
            settles=False,
        )
        test_households.ts.override_current("income", np.zeros(n_households))
        test_households.ts.override_current("expected_income", np.zeros(n_households))
        test_households.ts.override_current("liquid_financial_assets", np.full(n_households, 10.0))
        test_households.ts.override_current("received_consumption_loans", np.full(n_households, 50.0))
        test_households.ts.override_current("interest_paid", np.full(n_households, 5.0))
        test_households.ts.override_current("debt_installments", np.full(n_households, 10.0))
        _replace_post_grant_plan(
            test_households,
            credit_granted=np.full(n_households, 50.0),
            reserved_liquidation_total=np.zeros(n_households),
            settled_liquidation_total=np.zeros(n_households),
        )
        monkeypatch.setattr(
            test_households,
            "settle_post_grant_liquidation",
            lambda *, base_lfa, base_ifa: (base_lfa, base_ifa),
        )

        test_households.update_wealth(housing_data=pd.DataFrame(), tau_cf=0.0)

        # 10 opening LFA + 50 committed grant - 10 principal - 5 interest.
        np.testing.assert_allclose(test_households.ts.current("liquid_financial_assets"), 45.0)
        np.testing.assert_allclose(test_households.ts.current("illiquid_financial_assets"), 50.0)

    @pytest.mark.parametrize(
        ("opening_lfa, granted_credit, liquidation, expenditure, principal, interest, expected_lfa, expected_ifa"),
        [
            pytest.param(100.0, 0.0, 0.0, 30.0, 0.0, 0.0, 70.0, 50.0, id="lfa-only"),
            pytest.param(0.0, 50.0, 0.0, 50.0, 0.0, 0.0, 0.0, 50.0, id="credit-only"),
            pytest.param(0.0, 0.0, 30.0, 30.0, 0.0, 0.0, 0.0, 20.0, id="liquidation-only"),
            pytest.param(20.0, 0.0, 0.0, 0.0, 10.0, 5.0, 5.0, 50.0, id="debt-service-only"),
            pytest.param(10.0, 15.0, 20.0, 10.0, 3.0, 2.0, 30.0, 30.0, id="mixed-financing"),
        ],
    )
    def test__resolver_financing_matrix_reconciles_closing_financial_stocks(
        self,
        test_households,
        monkeypatch,
        opening_lfa,
        granted_credit,
        liquidation,
        expenditure,
        principal,
        interest,
        expected_lfa,
        expected_ifa,
    ):
        n_households, _ = self._configure_update_wealth(
            test_households,
            monkeypatch,
            resolver=True,
            settles=False,
            use_real_liquidation=True,
        )
        zeros = np.zeros(n_households)
        test_households.ts.override_current("income", zeros)
        test_households.ts.override_current("expected_income", zeros)
        test_households.ts.override_current("liquid_financial_assets", np.full(n_households, opening_lfa))
        test_households.ts.override_current("received_consumption_loans", np.full(n_households, granted_credit))
        test_households.ts.override_current("debt_installments", np.full(n_households, principal))
        test_households.ts.override_current("interest_paid", np.full(n_households, interest))
        spending = np.zeros_like(test_households.ts.current("nominal_amount_spent_in_lcu"))
        spending[:, 0] = expenditure
        test_households.ts.override_current("nominal_amount_spent_in_lcu", spending)
        _replace_post_grant_plan(
            test_households,
            credit_granted=np.full(n_households, granted_credit),
            funded_from_liquid_assets=np.full(n_households, min(opening_lfa, expenditure + principal + interest)),
            planned_liquidation_total=np.full(n_households, liquidation),
            reserved_liquidation_total=np.full(n_households, liquidation),
        )

        test_households.update_wealth(housing_data=pd.DataFrame(), tau_cf=0.0)

        np.testing.assert_allclose(test_households.ts.current("liquid_financial_assets"), expected_lfa)
        np.testing.assert_allclose(test_households.ts.current("illiquid_financial_assets"), expected_ifa)
        np.testing.assert_allclose(test_households.ts.current("wealth_financial_assets"), expected_lfa + expected_ifa)
        np.testing.assert_allclose(test_households.ts.current("stage5_cash_ledger_residual"), 0.0)

    def test__resolver_rejects_a_cash_grant_that_does_not_match_early_origination(
        self,
        test_households,
        monkeypatch,
    ):
        n_households, _ = self._configure_update_wealth(
            test_households,
            monkeypatch,
            resolver=True,
            settles=False,
        )
        test_households.ts.override_current("received_consumption_loans", np.full(n_households, 50.0))
        _replace_post_grant_plan(test_households, credit_granted=np.full(n_households, 49.0))
        initial_lfa_length = len(test_households.ts.dicts["liquid_financial_assets"])
        initial_ifa_length = len(test_households.ts.dicts["illiquid_financial_assets"])

        with pytest.raises(RuntimeError, match="received_consumption_loans to reconcile"):
            test_households.update_wealth(housing_data=pd.DataFrame(), tau_cf=0.0)

        assert len(test_households.ts.dicts["liquid_financial_assets"]) == initial_lfa_length
        assert len(test_households.ts.dicts["illiquid_financial_assets"]) == initial_ifa_length

    def test__settled_update_runs_actual_stage4_and_settlement_blocks(
        self,
        test_households,
        monkeypatch,
    ):
        n_households, settlement_calls = self._configure_update_wealth(
            test_households,
            monkeypatch,
            resolver=True,
            settles=True,
            use_actual_diagnostics=True,
        )

        test_households.update_wealth(housing_data=pd.DataFrame(), tau_cf=0.0)

        assert len(settlement_calls) == 1
        np.testing.assert_array_equal(
            test_households.ts.current("portfolio_settlement_enabled"),
            np.ones(n_households, dtype=bool),
        )
        expected_valid = ~test_households.ts.current("portfolio_no_financial_assets_flag")
        np.testing.assert_array_equal(test_households.ts.current("portfolio_settlement_valid_flag"), expected_valid)
        assert expected_valid.any()
        np.testing.assert_allclose(
            test_households.ts.current("wealth_financial_assets"),
            test_households.ts.current("liquid_financial_assets")
            + test_households.ts.current("illiquid_financial_assets"),
        )
        np.testing.assert_allclose(
            test_households.ts.current("portfolio_counterfactual_lfa_flow")
            + test_households.ts.current("portfolio_counterfactual_ifa_flow")
            + test_households.ts.current("portfolio_adjustment_cost"),
            np.zeros(n_households),
        )

    def test__legacy_compatibility_disabled_settlement_preserves_shadow_stock_update(
        self, test_households, monkeypatch
    ):
        n_households, settlement_calls = self._configure_update_wealth(
            test_households,
            monkeypatch,
            resolver=False,
            settles=False,
        )

        test_households.update_wealth(housing_data=pd.DataFrame(), tau_cf=0.0)

        assert settlement_calls == []
        np.testing.assert_allclose(test_households.ts.current("liquid_financial_assets"), np.full(n_households, 100.0))
        np.testing.assert_allclose(
            test_households.ts.current("illiquid_financial_assets"),
            np.full(n_households, 50.0),
        )
        np.testing.assert_allclose(test_households.ts.current("portfolio_settlement_enabled"), np.zeros(n_households))
        np.testing.assert_allclose(test_households.ts.current("portfolio_settlement_status"), np.zeros(n_households))

    def test__settled_update_rejects_invalid_stage5_authority_before_persistence(self, test_households, monkeypatch):
        n_households, _ = self._configure_update_wealth(
            test_households,
            monkeypatch,
            resolver=True,
            settles=True,
        )
        _replace_post_grant_plan(test_households, reserved_liquidation_total=np.full(n_households, np.nan))
        initial_lfa_length = len(test_households.ts.dicts["liquid_financial_assets"])
        initial_ifa_length = len(test_households.ts.dicts["illiquid_financial_assets"])
        initial_real_length = len(test_households.ts.dicts["wealth_real_assets"])

        with pytest.raises(RuntimeError, match="liquidation reservation must be a finite household vector"):
            test_households.update_wealth(housing_data=pd.DataFrame(), tau_cf=0.0)

        assert len(test_households.ts.dicts["liquid_financial_assets"]) == initial_lfa_length
        assert len(test_households.ts.dicts["illiquid_financial_assets"]) == initial_ifa_length
        assert len(test_households.ts.dicts["wealth_real_assets"]) == initial_real_length

    def test__settled_update_rejects_inconsistent_stage5_authority(self, test_households, monkeypatch):
        n_households, _ = self._configure_update_wealth(
            test_households,
            monkeypatch,
            resolver=True,
            settles=True,
        )
        _replace_post_grant_plan(test_households, reserved_liquidation_total=np.full(n_households, 51.0))

        with pytest.raises(RuntimeError, match="cannot be honoured before return settlement"):
            test_households.update_wealth(housing_data=pd.DataFrame(), tau_cf=0.0)


class TestComputeAndRecordLiquidityShortfall:
    """Stage 5 (feasibility resolver) Increment 0: liquidity-shortfall diagnostic."""

    def test__defaults_to_expected_income_when_no_override_supplied(self, test_households):
        n_households = test_households.ts.current("n_households")
        n_industries = test_households.n_industries
        test_households.ts.override_current("income", np.full(n_households, 111.0))
        test_households.ts.override_current("expected_income", np.full(n_households, 222.0))
        # Isolate the income basis under test: cash rent is a separate use of
        # the period's resources (GH #120) and is covered by its own test below.
        test_households.ts.override_current("rent", np.zeros(n_households))

        shortfall = test_households.compute_and_record_liquidity_shortfall(
            target_consumption=np.zeros((n_households, n_industries)),
            scheduled_debt_service=np.zeros(n_households),
        )

        # target_consumption, cash rent, and scheduled_debt_service are all
        # zero, so liquidity_shortfall == -income; must reflect expected_income
        # (222), not income (111). See round-2 review finding in households.py's
        # compute_and_record_liquidity_shortfall docstring.
        np.testing.assert_allclose(shortfall, np.full(n_households, -222.0))

    def test__income_override_takes_precedence_over_expected_income(self, test_households):
        # Mirrors compute_target_consumption's own income_override semantics
        # (test__target_consumption_uses_income_override above) so this
        # diagnostic cannot silently diverge from compute_target_consumption's
        # income basis if a future caller starts passing an override through
        # _set_household_target_demand (round-3 review finding: pre-empts the
        # same class of bug round 2 found, before it becomes live).
        n_households = test_households.ts.current("n_households")
        n_industries = test_households.n_industries
        test_households.ts.override_current("expected_income", np.full(n_households, 222.0))
        test_households.ts.override_current("rent", np.zeros(n_households))
        income_override = np.full(n_households, 999.0)

        shortfall = test_households.compute_and_record_liquidity_shortfall(
            target_consumption=np.zeros((n_households, n_industries)),
            scheduled_debt_service=np.zeros(n_households),
            income_override=income_override,
        )

        np.testing.assert_allclose(shortfall, np.full(n_households, -999.0))

    def test__cash_rent_counts_as_a_use_but_imputed_rent_never_does(self, test_households):
        # GH #120: target_consumption now carries market expenditure only, so
        # actual rent must enter the shortfall as its own cash use. Imputed rent
        # is measured consumption, not a liability, and must never move the
        # shortfall -- otherwise owner-occupiers would draw feasibility support
        # for a payment nobody is owed.
        n_households = test_households.ts.current("n_households")
        n_industries = test_households.n_industries
        test_households.ts.override_current("expected_income", np.full(n_households, 100.0))
        test_households.ts.override_current("rent", np.full(n_households, 30.0))
        test_households.ts.override_current("rent_imputed", np.full(n_households, 500.0))

        shortfall = test_households.compute_and_record_liquidity_shortfall(
            target_consumption=np.zeros((n_households, n_industries)),
            scheduled_debt_service=np.zeros(n_households),
        )

        # 30 (cash rent) - 100 (income) = -70. The 500 of imputed rent is absent.
        np.testing.assert_allclose(shortfall, np.full(n_households, -70.0))
        np.testing.assert_allclose(test_households.ts.current("household_saving"), np.full(n_households, 70.0))


class TestComputeAndRecordLiquidAssetDrawdown:
    """Stage 5 (feasibility resolver) Increment 1: liquid-asset drawdown diagnostic."""

    def test__liquid_asset_drawdown_appends_diagnostics_using_current_liquid_financial_assets(self, test_households):
        n_households = test_households.ts.current("n_households")
        deposits = np.resize(np.asarray([50.0, 20.0, -5.0]), n_households)
        liquidity_shortfall = np.resize(np.asarray([100.0, 10.0, 30.0]), n_households)
        expected_funded = np.resize(np.asarray([50.0, 10.0, 0.0]), n_households)
        expected_residual = np.resize(np.asarray([50.0, 0.0, 30.0]), n_households)
        test_households.ts.override_current("liquid_financial_assets", deposits)

        residual = test_households.compute_and_record_liquid_asset_drawdown(liquidity_shortfall)

        np.testing.assert_allclose(
            test_households.ts.current("liquidity_shortfall_before_repair"),
            liquidity_shortfall,
        )
        np.testing.assert_allclose(
            test_households.ts.current("funded_from_liquid_assets"),
            expected_funded,
        )
        np.testing.assert_allclose(
            test_households.ts.current("residual_shortfall_after_lfa"),
            expected_residual,
        )
        np.testing.assert_allclose(residual, test_households.ts.current("residual_shortfall_after_lfa"))

    def test__liquid_asset_drawdown_replace_current_overrides_latest_diagnostics(self, test_households):
        n_households = test_households.ts.current("n_households")
        first_shortfall = np.full(n_households, 10.0)
        second_shortfall = np.full(n_households, 25.0)
        test_households.ts.override_current("liquid_financial_assets", np.full(n_households, 5.0))

        test_households.compute_and_record_liquid_asset_drawdown(first_shortfall)
        before_lengths = {
            key: len(test_households.ts.dicts[key])
            for key in [
                "liquidity_shortfall_before_repair",
                "funded_from_liquid_assets",
                "residual_shortfall_after_lfa",
            ]
        }

        test_households.compute_and_record_liquid_asset_drawdown(second_shortfall, replace_current=True)

        after_lengths = {key: len(test_households.ts.dicts[key]) for key in before_lengths}
        assert after_lengths == before_lengths
        np.testing.assert_allclose(test_households.ts.current("liquidity_shortfall_before_repair"), second_shortfall)
        np.testing.assert_allclose(test_households.ts.current("funded_from_liquid_assets"), np.full(n_households, 5.0))
        np.testing.assert_allclose(
            test_households.ts.current("residual_shortfall_after_lfa"),
            np.full(n_households, 20.0),
        )

    def test__liquid_asset_drawdown_non_finite_shortfall_snapshot_is_recorded_as_zero(self, test_households):
        n_households = test_households.ts.current("n_households")
        test_households.ts.override_current("liquid_financial_assets", np.full(n_households, 10.0))
        liquidity_shortfall = np.resize(np.asarray([np.nan, np.inf, -8.0, 5.0]), n_households)
        expected_snapshot = np.resize(np.asarray([0.0, 0.0, 0.0, 5.0]), n_households)

        test_households.compute_and_record_liquid_asset_drawdown(liquidity_shortfall)

        np.testing.assert_allclose(
            test_households.ts.current("liquidity_shortfall_before_repair"),
            expected_snapshot,
        )
        np.testing.assert_allclose(
            test_households.ts.current("funded_from_liquid_assets")
            + test_households.ts.current("residual_shortfall_after_lfa"),
            expected_snapshot,
        )

    def test__populate_pre_grant_feasible_plan_from_liquid_asset_drawdown_copies_shadow_values(self, test_households):
        n_households = test_households.ts.current("n_households")
        liquidity_shortfall = np.resize(np.asarray([15.0, np.nan, -2.0]), n_households)
        test_households.ts.override_current("liquid_financial_assets", np.full(n_households, 10.0))
        test_households.configure_feasibility_resolver(True)

        expected_deposits = test_households.ts.current("liquid_financial_assets").copy()
        expected_financial_assets = test_households.ts.current("wealth_financial_assets").copy()
        expected_loans = test_households.ts.current("target_consumption_loans").copy()
        residual = test_households.compute_and_record_liquid_asset_drawdown(liquidity_shortfall)
        test_households.populate_pre_grant_feasible_plan_from_liquid_asset_drawdown(
            liquidity_shortfall_before_repair=test_households.ts.current("liquidity_shortfall_before_repair"),
            funded_from_liquid_assets=test_households.ts.current("funded_from_liquid_assets"),
            residual_shortfall_after_lfa=residual,
        )

        assert test_households.pre_grant_feasible_plan is not None
        np.testing.assert_allclose(
            test_households.pre_grant_feasible_plan.liquidity_shortfall_before_repair,
            test_households.ts.current("liquidity_shortfall_before_repair"),
        )
        np.testing.assert_allclose(
            test_households.pre_grant_feasible_plan.funded_from_liquid_assets,
            test_households.ts.current("funded_from_liquid_assets"),
        )
        np.testing.assert_allclose(
            test_households.current_live_post_drawdown_residual(),
            test_households.ts.current("residual_shortfall_after_lfa"),
        )
        np.testing.assert_allclose(test_households.ts.current("liquid_financial_assets"), expected_deposits)
        np.testing.assert_allclose(
            test_households.ts.current("wealth_financial_assets"),
            expected_financial_assets,
        )
        np.testing.assert_allclose(
            test_households.ts.current("target_consumption_loans"),
            expected_loans,
            equal_nan=True,
        )

    def test__configure_feasibility_resolver_clears_stale_pre_grant_feasible_plan(self, test_households):
        n_households = test_households.ts.current("n_households")
        test_households.populate_pre_grant_feasible_plan_from_liquid_asset_drawdown(
            liquidity_shortfall_before_repair=np.full(n_households, 3.0),
            funded_from_liquid_assets=np.full(n_households, 1.0),
            residual_shortfall_after_lfa=np.full(n_households, 2.0),
        )
        test_households.post_grant_feasible_plan = households_module.PostGrantFeasiblePlan(
            credit_granted=np.full(n_households, 2.0),
            credit_rationing_gap=np.zeros(n_households),
            planned_liquidation_total=np.zeros(n_households),
            residual_shortfall_after_granted_credit=np.zeros(n_households),
        )

        test_households.configure_feasibility_resolver(True)

        assert test_households.pre_grant_feasible_plan is None
        assert test_households.post_grant_feasible_plan is None

    def test__configure_feasibility_resolver_can_preserve_settled_post_grant_carrier(self, test_households):
        n_households = test_households.ts.current("n_households")
        test_households.populate_pre_grant_feasible_plan_from_liquid_asset_drawdown(
            liquidity_shortfall_before_repair=np.full(n_households, 3.0),
            funded_from_liquid_assets=np.full(n_households, 1.0),
            residual_shortfall_after_lfa=np.full(n_households, 2.0),
        )
        post_grant_plan = households_module.PostGrantFeasiblePlan(
            credit_granted=np.full(n_households, 2.0),
            credit_rationing_gap=np.zeros(n_households),
            planned_liquidation_total=np.zeros(n_households),
            residual_shortfall_after_granted_credit=np.zeros(n_households),
        )
        test_households.post_grant_feasible_plan = post_grant_plan

        test_households.configure_feasibility_resolver(True, clear_post_grant=False)

        assert test_households.pre_grant_feasible_plan is None
        assert test_households.post_grant_feasible_plan is post_grant_plan

    def test__configure_feasibility_resolver_always_clears_post_grant_when_disabled(self, test_households):
        n_households = test_households.ts.current("n_households")
        test_households.post_grant_feasible_plan = households_module.PostGrantFeasiblePlan(
            credit_granted=np.full(n_households, 2.0),
            credit_rationing_gap=np.zeros(n_households),
            planned_liquidation_total=np.zeros(n_households),
            residual_shortfall_after_granted_credit=np.zeros(n_households),
        )

        test_households.configure_feasibility_resolver(False, clear_post_grant=False)

        assert test_households.post_grant_feasible_plan is None

    def test__current_live_post_drawdown_residual_raises_when_enabled_without_live_carrier(self, test_households):
        test_households.configure_feasibility_resolver(True)

        with pytest.raises(RuntimeError, match="pre_grant_feasible_plan"):
            test_households.current_live_post_drawdown_residual()

    def test__reset_with_resolver_enabled_clears_stale_pre_grant_feasible_plan(self, test_households):
        n_households = test_households.ts.current("n_households")
        configuration = HouseholdsConfiguration()
        configuration.parameters.uses_feasibility_resolver = True
        test_households.populate_pre_grant_feasible_plan_from_liquid_asset_drawdown(
            liquidity_shortfall_before_repair=np.full(n_households, 3.0),
            funded_from_liquid_assets=np.full(n_households, 1.0),
            residual_shortfall_after_lfa=np.full(n_households, 2.0),
        )

        test_households.reset(configuration)

        assert test_households.uses_feasibility_resolver is True
        assert test_households.pre_grant_feasible_plan is None

    def test__legacy_compatibility_post_drawdown_residual_falls_back_to_clipped_shadow_value(self, test_households):
        n_households = test_households.ts.current("n_households")
        residual_shortfall_after_lfa = np.resize(np.asarray([12.0, -3.0, np.nan, np.inf]), n_households)
        expected = np.resize(np.asarray([12.0, 0.0, 0.0, 0.0]), n_households)
        test_households.ts.override_current("residual_shortfall_after_lfa", residual_shortfall_after_lfa)
        test_households.configure_feasibility_resolver(False)

        np.testing.assert_allclose(test_households.current_live_post_drawdown_residual(), expected)


class TestPopulateAndAccessLiveCreditRequested:
    """Stage 5 (feasibility resolver) Increment 5: live credit_requested handoff."""

    def test__populate_pre_grant_feasible_plan_credit_requested_raises_without_existing_carrier(self, test_households):
        n_households = test_households.ts.current("n_households")
        test_households.configure_feasibility_resolver(True)

        with pytest.raises(RuntimeError, match="pre_grant_feasible_plan"):
            test_households.populate_pre_grant_feasible_plan_credit_requested(
                credit_requested=np.full(n_households, 5.0)
            )

    def test__populate_pre_grant_feasible_plan_credit_requested_extends_existing_carrier_in_place(
        self, test_households
    ):
        n_households = test_households.ts.current("n_households")
        test_households.populate_pre_grant_feasible_plan_from_liquid_asset_drawdown(
            liquidity_shortfall_before_repair=np.full(n_households, 3.0),
            funded_from_liquid_assets=np.full(n_households, 1.0),
            residual_shortfall_after_lfa=np.full(n_households, 2.0),
        )
        credit_requested = np.full(n_households, 7.0)

        test_households.populate_pre_grant_feasible_plan_credit_requested(credit_requested=credit_requested)

        plan = test_households.pre_grant_feasible_plan
        assert plan is not None
        np.testing.assert_allclose(plan.credit_requested, credit_requested)
        # Increment 4's fields must be untouched by this additive extension.
        np.testing.assert_allclose(plan.liquidity_shortfall_before_repair, np.full(n_households, 3.0))
        np.testing.assert_allclose(plan.funded_from_liquid_assets, np.full(n_households, 1.0))
        np.testing.assert_allclose(plan.residual_shortfall_after_lfa, np.full(n_households, 2.0))

    def test__current_live_credit_requested_raises_when_enabled_without_carrier(self, test_households):
        test_households.configure_feasibility_resolver(True)

        with pytest.raises(RuntimeError, match="pre_grant_feasible_plan"):
            test_households.current_live_credit_requested()

    def test__current_live_credit_requested_raises_distinctly_when_carrier_missing_field(self, test_households):
        n_households = test_households.ts.current("n_households")
        test_households.configure_feasibility_resolver(True)
        test_households.populate_pre_grant_feasible_plan_from_liquid_asset_drawdown(
            liquidity_shortfall_before_repair=np.full(n_households, 3.0),
            funded_from_liquid_assets=np.full(n_households, 1.0),
            residual_shortfall_after_lfa=np.full(n_households, 2.0),
        )

        with pytest.raises(RuntimeError, match="credit_requested has not been populated"):
            test_households.current_live_credit_requested()

    def test__current_live_credit_requested_returns_carrier_value_when_enabled(self, test_households):
        n_households = test_households.ts.current("n_households")
        test_households.configure_feasibility_resolver(True)
        test_households.populate_pre_grant_feasible_plan_from_liquid_asset_drawdown(
            liquidity_shortfall_before_repair=np.full(n_households, 3.0),
            funded_from_liquid_assets=np.full(n_households, 1.0),
            residual_shortfall_after_lfa=np.full(n_households, 2.0),
        )
        credit_requested = np.resize(np.asarray([9.0, -1.0, np.nan]), n_households)
        expected = np.resize(np.asarray([9.0, 0.0, 0.0]), n_households)
        test_households.populate_pre_grant_feasible_plan_credit_requested(credit_requested=credit_requested)

        np.testing.assert_allclose(test_households.current_live_credit_requested(), expected)

    def test__legacy_compatibility_credit_request_falls_back_to_shadow_value(self, test_households):
        n_households = test_households.ts.current("n_households")
        shadow = np.resize(np.asarray([11.0, -2.0, np.inf]), n_households)
        expected = np.resize(np.asarray([11.0, 0.0, 0.0]), n_households)
        test_households.ts.override_current("shadow_credit_requested", shadow)
        test_households.configure_feasibility_resolver(False)

        np.testing.assert_allclose(test_households.current_live_credit_requested(), expected)


class TestPopulateAndAccessLivePlannedLiquidation:
    """Stage 5 (feasibility resolver) Increment 6: live liquidation handoff."""

    def test__populate_pre_grant_feasible_plan_planned_liquidation_raises_without_carrier(self, test_households):
        n_households = test_households.ts.current("n_households")
        test_households.configure_feasibility_resolver(True)

        with pytest.raises(RuntimeError, match="pre_grant_feasible_plan"):
            test_households.populate_pre_grant_feasible_plan_planned_liquidation(
                planned_liquidation_total=np.full(n_households, 5.0),
                current_ifa=np.full(n_households, 10.0),
            )

    def test__populate_pre_grant_feasible_plan_planned_liquidation_extends_existing_carrier(self, test_households):
        n_households = test_households.ts.current("n_households")
        credit_requested = np.full(n_households, 7.0)
        planned_liquidation = np.full(n_households, 4.0)
        test_households.populate_pre_grant_feasible_plan_from_liquid_asset_drawdown(
            liquidity_shortfall_before_repair=np.full(n_households, 3.0),
            funded_from_liquid_assets=np.full(n_households, 1.0),
            residual_shortfall_after_lfa=np.full(n_households, 2.0),
        )
        test_households.populate_pre_grant_feasible_plan_credit_requested(credit_requested=credit_requested)

        test_households.populate_pre_grant_feasible_plan_planned_liquidation(
            planned_liquidation_total=planned_liquidation,
            current_ifa=np.full(n_households, 10.0),
        )

        plan = test_households.pre_grant_feasible_plan
        assert plan is not None
        np.testing.assert_allclose(plan.planned_liquidation_total, planned_liquidation)
        np.testing.assert_allclose(plan.credit_requested, credit_requested)
        np.testing.assert_allclose(plan.liquidity_shortfall_before_repair, np.full(n_households, 3.0))
        np.testing.assert_allclose(plan.funded_from_liquid_assets, np.full(n_households, 1.0))
        np.testing.assert_allclose(plan.residual_shortfall_after_lfa, np.full(n_households, 2.0))

    def test__populate_pre_grant_feasible_plan_planned_liquidation_clamps_invalid_values(self, test_households):
        n_households = test_households.ts.current("n_households")
        planned_liquidation = np.resize(np.asarray([5.0, -2.0, np.nan, np.inf, 50.0]), n_households)
        current_ifa = np.resize(np.asarray([10.0, 10.0, 10.0, 10.0, 12.0]), n_households)
        expected = np.resize(np.asarray([5.0, 0.0, 0.0, 0.0, 12.0]), n_households)
        test_households.populate_pre_grant_feasible_plan_from_liquid_asset_drawdown(
            liquidity_shortfall_before_repair=np.zeros(n_households),
            funded_from_liquid_assets=np.zeros(n_households),
            residual_shortfall_after_lfa=np.zeros(n_households),
        )

        test_households.populate_pre_grant_feasible_plan_planned_liquidation(
            planned_liquidation_total=planned_liquidation,
            current_ifa=current_ifa,
        )

        np.testing.assert_allclose(test_households.pre_grant_feasible_plan.planned_liquidation_total, expected)

    @pytest.mark.parametrize(
        "planned_liquidation",
        [
            np.asarray(5.0),
            np.asarray([1.0, 2.0]),
            np.asarray([[1.0], [2.0], [3.0]]),
        ],
    )
    def test__populate_pre_grant_feasible_plan_planned_liquidation_rejects_bad_liquidation_shape(
        self, test_households, planned_liquidation
    ):
        n_households = test_households.ts.current("n_households")
        test_households.populate_pre_grant_feasible_plan_from_liquid_asset_drawdown(
            liquidity_shortfall_before_repair=np.zeros(n_households),
            funded_from_liquid_assets=np.zeros(n_households),
            residual_shortfall_after_lfa=np.zeros(n_households),
        )

        with pytest.raises(ValueError, match="planned_liquidation_total must contain exactly one value per household"):
            test_households.populate_pre_grant_feasible_plan_planned_liquidation(
                planned_liquidation_total=planned_liquidation,
                current_ifa=np.full(n_households, 10.0),
            )

    def test__populate_pre_grant_feasible_plan_planned_liquidation_rejects_bad_ifa_shape(self, test_households):
        n_households = test_households.ts.current("n_households")
        test_households.populate_pre_grant_feasible_plan_from_liquid_asset_drawdown(
            liquidity_shortfall_before_repair=np.zeros(n_households),
            funded_from_liquid_assets=np.zeros(n_households),
            residual_shortfall_after_lfa=np.zeros(n_households),
        )

        with pytest.raises(ValueError, match="current_ifa must contain exactly one value per household"):
            test_households.populate_pre_grant_feasible_plan_planned_liquidation(
                planned_liquidation_total=np.full(n_households, 5.0),
                current_ifa=np.asarray([[10.0], [10.0], [10.0]]),
            )

    def test__populate_pre_grant_feasible_plan_planned_liquidation_copies_inputs(self, test_households):
        n_households = test_households.ts.current("n_households")
        planned_liquidation = np.full(n_households, 5.0)
        test_households.populate_pre_grant_feasible_plan_from_liquid_asset_drawdown(
            liquidity_shortfall_before_repair=np.full(n_households, 3.0),
            funded_from_liquid_assets=np.full(n_households, 1.0),
            residual_shortfall_after_lfa=np.full(n_households, 2.0),
        )

        test_households.populate_pre_grant_feasible_plan_planned_liquidation(
            planned_liquidation_total=planned_liquidation,
            current_ifa=np.full(n_households, 10.0),
        )
        planned_liquidation[:] = 99.0

        np.testing.assert_allclose(
            test_households.pre_grant_feasible_plan.planned_liquidation_total,
            np.full(n_households, 5.0),
        )

    def test__current_live_planned_liquidation_total_raises_when_enabled_without_carrier(self, test_households):
        test_households.configure_feasibility_resolver(True)

        with pytest.raises(RuntimeError, match="pre_grant_feasible_plan"):
            test_households.current_live_planned_liquidation_total()

    def test__current_live_planned_liquidation_total_raises_when_enabled_without_populated_field(self, test_households):
        n_households = test_households.ts.current("n_households")
        test_households.configure_feasibility_resolver(True)
        test_households.populate_pre_grant_feasible_plan_from_liquid_asset_drawdown(
            liquidity_shortfall_before_repair=np.zeros(n_households),
            funded_from_liquid_assets=np.zeros(n_households),
            residual_shortfall_after_lfa=np.zeros(n_households),
        )

        with pytest.raises(RuntimeError, match="planned_liquidation_total has not been populated"):
            test_households.current_live_planned_liquidation_total()

    def test__current_live_planned_liquidation_total_returns_carrier_value_when_enabled(self, test_households):
        n_households = test_households.ts.current("n_households")
        test_households.configure_feasibility_resolver(True)
        test_households.populate_pre_grant_feasible_plan_from_liquid_asset_drawdown(
            liquidity_shortfall_before_repair=np.zeros(n_households),
            funded_from_liquid_assets=np.zeros(n_households),
            residual_shortfall_after_lfa=np.zeros(n_households),
        )
        test_households.populate_pre_grant_feasible_plan_planned_liquidation(
            planned_liquidation_total=np.full(n_households, 6.0),
            current_ifa=np.full(n_households, 10.0),
        )

        np.testing.assert_allclose(
            test_households.current_live_planned_liquidation_total(),
            np.full(n_households, 6.0),
        )

    def test__legacy_compatibility_planned_liquidation_falls_back_to_shadow_value(self, test_households):
        n_households = test_households.ts.current("n_households")
        shadow = np.resize(np.asarray([11.0, -2.0, np.inf]), n_households)
        expected = np.resize(np.asarray([11.0, 0.0, 0.0]), n_households)
        test_households.ts.override_current("liquidation_planned", shadow)
        test_households.configure_feasibility_resolver(False)

        np.testing.assert_allclose(test_households.current_live_planned_liquidation_total(), expected)


class TestPopulatePostGrantFeasiblePlan:
    """Stage 5 Increment 7: settled post-grant feasibility carrier."""

    @staticmethod
    def _populate_pre_grant_plan(
        test_households,
        *,
        residual_after_lfa,
        credit_requested,
        planned_liquidation,
    ):
        n_households = test_households.ts.current("n_households")
        test_households.configure_feasibility_resolver(True)
        test_households.populate_pre_grant_feasible_plan_from_liquid_asset_drawdown(
            liquidity_shortfall_before_repair=np.asarray(residual_after_lfa, dtype=float) + 5.0,
            funded_from_liquid_assets=np.full(n_households, 5.0),
            residual_shortfall_after_lfa=residual_after_lfa,
        )
        test_households.populate_pre_grant_feasible_plan_credit_requested(credit_requested=credit_requested)
        test_households.populate_pre_grant_feasible_plan_planned_liquidation(
            planned_liquidation_total=planned_liquidation,
            current_ifa=np.full(n_households, 100.0),
        )

    def test__full_grant_builds_settled_carrier_without_rationing(self, test_households):
        n_households = test_households.ts.current("n_households")
        credit_requested = np.full(n_households, 8.0)
        planned_liquidation = np.full(n_households, 4.0)
        self._populate_pre_grant_plan(
            test_households,
            residual_after_lfa=np.full(n_households, 12.0),
            credit_requested=credit_requested,
            planned_liquidation=planned_liquidation,
        )

        test_households.populate_post_grant_feasible_plan_from_granted_credit(credit_granted=credit_requested)

        plan = test_households.post_grant_feasible_plan
        assert plan is not None
        np.testing.assert_allclose(plan.credit_granted, credit_requested)
        np.testing.assert_allclose(plan.credit_rationing_gap, np.zeros(n_households))
        np.testing.assert_allclose(plan.planned_liquidation_total, planned_liquidation)
        np.testing.assert_allclose(plan.residual_shortfall_after_granted_credit, np.zeros(n_households))

    def test__partial_grant_preserves_rationing_gap_and_residual(self, test_households):
        n_households = test_households.ts.current("n_households")
        self._populate_pre_grant_plan(
            test_households,
            residual_after_lfa=np.full(n_households, 20.0),
            credit_requested=np.full(n_households, 12.0),
            planned_liquidation=np.full(n_households, 3.0),
        )

        test_households.populate_post_grant_feasible_plan_from_granted_credit(
            credit_granted=np.full(n_households, 5.0),
        )

        plan = test_households.post_grant_feasible_plan
        np.testing.assert_allclose(plan.credit_granted, np.full(n_households, 5.0))
        np.testing.assert_allclose(plan.credit_rationing_gap, np.full(n_households, 7.0))
        np.testing.assert_allclose(plan.residual_shortfall_after_granted_credit, np.full(n_households, 12.0))

    def test__zero_grant_is_explicit_and_never_falls_back_to_requested_credit(self, test_households):
        n_households = test_households.ts.current("n_households")
        self._populate_pre_grant_plan(
            test_households,
            residual_after_lfa=np.full(n_households, 10.0),
            credit_requested=np.full(n_households, 6.0),
            planned_liquidation=np.full(n_households, 1.0),
        )

        test_households.populate_post_grant_feasible_plan_from_granted_credit(
            credit_granted=np.zeros(n_households),
        )

        plan = test_households.post_grant_feasible_plan
        np.testing.assert_allclose(plan.credit_granted, np.zeros(n_households))
        np.testing.assert_allclose(plan.credit_rationing_gap, np.full(n_households, 6.0))
        np.testing.assert_allclose(plan.residual_shortfall_after_granted_credit, np.full(n_households, 9.0))

    def test__post_grant_carrier_reconciles_household_and_bank_settlement(self, test_households):
        n_households = test_households.ts.current("n_households")
        credit_granted = np.arange(1.0, n_households + 1.0)
        settlement = np.zeros((2, n_households))
        settlement[0] = credit_granted * 0.25
        settlement[1] = credit_granted * 0.75
        self._populate_pre_grant_plan(
            test_households,
            residual_after_lfa=credit_granted,
            credit_requested=credit_granted,
            planned_liquidation=np.zeros(n_households),
        )

        test_households.populate_post_grant_feasible_plan_from_granted_credit(
            credit_granted=credit_granted,
            granted_consumer_credit_by_bank_and_household=settlement,
        )

        plan = test_households.post_grant_feasible_plan
        np.testing.assert_allclose(plan.granted_consumer_credit_by_bank_and_household, settlement)
        np.testing.assert_allclose(plan.consumer_debt_liability_booking, credit_granted)
        np.testing.assert_allclose(plan.bank_consumer_loan_asset_booking, settlement.sum(axis=1))

    def test__post_grant_carrier_rejects_non_reconciling_bank_settlement(self, test_households):
        n_households = test_households.ts.current("n_households")
        self._populate_pre_grant_plan(
            test_households,
            residual_after_lfa=np.ones(n_households),
            credit_requested=np.ones(n_households),
            planned_liquidation=np.zeros(n_households),
        )

        with pytest.raises(RuntimeError, match="does not reconcile"):
            test_households.populate_post_grant_feasible_plan_from_granted_credit(
                credit_granted=np.ones(n_households),
                granted_consumer_credit_by_bank_and_household=np.zeros((2, n_households)),
            )

    def test__post_grant_reconciliation_leaves_pre_grant_carrier_unchanged(self, test_households):
        n_households = test_households.ts.current("n_households")
        self._populate_pre_grant_plan(
            test_households,
            residual_after_lfa=np.full(n_households, 10.0),
            credit_requested=np.full(n_households, 6.0),
            planned_liquidation=np.full(n_households, 2.0),
        )
        pre_grant_plan = test_households.pre_grant_feasible_plan
        original_credit_requested = pre_grant_plan.credit_requested.copy()
        original_planned_liquidation = pre_grant_plan.planned_liquidation_total.copy()

        test_households.populate_post_grant_feasible_plan_from_granted_credit(
            credit_granted=np.full(n_households, 4.0),
        )

        np.testing.assert_allclose(pre_grant_plan.credit_requested, original_credit_requested)
        np.testing.assert_allclose(pre_grant_plan.planned_liquidation_total, original_planned_liquidation)

    def test__post_grant_carrier_does_not_alias_pre_grant_arrays_or_inputs(self, test_households):
        n_households = test_households.ts.current("n_households")
        credit_granted = np.full(n_households, 4.0)
        self._populate_pre_grant_plan(
            test_households,
            residual_after_lfa=np.full(n_households, 10.0),
            credit_requested=np.full(n_households, 6.0),
            planned_liquidation=np.full(n_households, 2.0),
        )

        test_households.populate_post_grant_feasible_plan_from_granted_credit(credit_granted=credit_granted)
        credit_granted[:] = 99.0
        with pytest.raises(ValueError, match="read-only"):
            test_households.pre_grant_feasible_plan.planned_liquidation_total[:] = 88.0

        plan = test_households.post_grant_feasible_plan
        np.testing.assert_allclose(plan.credit_granted, np.full(n_households, 4.0))
        np.testing.assert_allclose(plan.planned_liquidation_total, np.full(n_households, 2.0))

    def test__persist_post_grant_planned_liquidation_total_copies_settled_values(self, test_households):
        n_households = test_households.ts.current("n_households")
        self._populate_pre_grant_plan(
            test_households,
            residual_after_lfa=np.full(n_households, 10.0),
            credit_requested=np.full(n_households, 6.0),
            planned_liquidation=np.full(n_households, 2.0),
        )
        test_households.populate_post_grant_feasible_plan_from_granted_credit(
            credit_granted=np.full(n_households, 4.0),
        )
        test_households.ts.override_current("liquidation_planned", np.full(n_households, 99.0))

        test_households.persist_post_grant_planned_liquidation_total()
        with pytest.raises(ValueError, match="read-only"):
            test_households.post_grant_feasible_plan.planned_liquidation_total[:] = 77.0

        np.testing.assert_allclose(
            test_households.ts.current("liquidation_planned"),
            np.full(n_households, 2.0),
        )

    def test__post_grant_liquidation_exposes_authoritative_bases_once(self, test_households):
        n_households = test_households.ts.current("n_households")
        self._populate_pre_grant_plan(
            test_households,
            residual_after_lfa=np.full(n_households, 10.0),
            credit_requested=np.full(n_households, 6.0),
            planned_liquidation=np.full(n_households, 2.0),
        )
        test_households.populate_post_grant_feasible_plan_from_granted_credit(
            credit_granted=np.full(n_households, 4.0),
        )

        post_lfa, post_ifa = test_households.settle_post_grant_liquidation(
            base_lfa=np.full(n_households, 100.0),
            base_ifa=np.full(n_households, 50.0),
        )

        np.testing.assert_allclose(post_lfa, np.full(n_households, 102.0))
        np.testing.assert_allclose(post_ifa, np.full(n_households, 48.0))
        np.testing.assert_allclose(
            test_households.post_grant_feasible_plan.residual_shortfall_after_granted_credit,
            np.full(n_households, 4.0),
        )
        with pytest.raises(RuntimeError, match="already been applied"):
            test_households.settle_post_grant_liquidation(
                base_lfa=np.full(n_households, 100.0),
                base_ifa=np.full(n_households, 50.0),
            )

    def test__post_grant_reconciliation_raises_without_pre_grant_carrier(self, test_households):
        n_households = test_households.ts.current("n_households")
        test_households.configure_feasibility_resolver(True)

        with pytest.raises(RuntimeError, match="pre_grant_feasible_plan"):
            test_households.populate_post_grant_feasible_plan_from_granted_credit(
                credit_granted=np.zeros(n_households),
            )

    def test__post_grant_reconciliation_raises_without_pre_grant_credit_requested(self, test_households):
        n_households = test_households.ts.current("n_households")
        test_households.configure_feasibility_resolver(True)
        test_households.populate_pre_grant_feasible_plan_from_liquid_asset_drawdown(
            liquidity_shortfall_before_repair=np.zeros(n_households),
            funded_from_liquid_assets=np.zeros(n_households),
            residual_shortfall_after_lfa=np.zeros(n_households),
        )

        with pytest.raises(RuntimeError, match="credit_requested"):
            test_households.populate_post_grant_feasible_plan_from_granted_credit(
                credit_granted=np.zeros(n_households),
            )

    def test__post_grant_reconciliation_raises_without_pre_grant_planned_liquidation(self, test_households):
        n_households = test_households.ts.current("n_households")
        test_households.configure_feasibility_resolver(True)
        test_households.populate_pre_grant_feasible_plan_from_liquid_asset_drawdown(
            liquidity_shortfall_before_repair=np.zeros(n_households),
            funded_from_liquid_assets=np.zeros(n_households),
            residual_shortfall_after_lfa=np.zeros(n_households),
        )
        test_households.populate_pre_grant_feasible_plan_credit_requested(
            credit_requested=np.zeros(n_households),
        )

        with pytest.raises(RuntimeError, match="planned_liquidation_total"):
            test_households.populate_post_grant_feasible_plan_from_granted_credit(
                credit_granted=np.zeros(n_households),
            )

    @pytest.mark.parametrize(
        "credit_granted",
        [
            np.asarray(0.0),
            np.asarray([1.0, 2.0]),
            np.asarray([[1.0], [2.0], [3.0]]),
        ],
    )
    def test__post_grant_reconciliation_rejects_bad_granted_credit_shape(self, test_households, credit_granted):
        n_households = test_households.ts.current("n_households")
        self._populate_pre_grant_plan(
            test_households,
            residual_after_lfa=np.zeros(n_households),
            credit_requested=np.zeros(n_households),
            planned_liquidation=np.zeros(n_households),
        )

        with pytest.raises(ValueError, match="credit_granted must contain exactly one value per household"):
            test_households.populate_post_grant_feasible_plan_from_granted_credit(
                credit_granted=credit_granted,
            )

    def test__post_grant_reconciliation_raises_on_non_finite_granted_credit(self, test_households):
        n_households = test_households.ts.current("n_households")
        self._populate_pre_grant_plan(
            test_households,
            residual_after_lfa=np.full(n_households, 10.0),
            credit_requested=np.full(n_households, 8.0),
            planned_liquidation=np.full(n_households, 2.0),
        )

        with pytest.raises(RuntimeError, match="finite cleared credit_granted"):
            test_households.populate_post_grant_feasible_plan_from_granted_credit(
                credit_granted=np.resize(np.asarray([5.0, np.nan, 4.0, np.inf]), n_households),
            )

    def test__post_grant_reconciliation_clamps_non_finite_pre_grant_values(self, test_households):
        n_households = test_households.ts.current("n_households")
        self._populate_pre_grant_plan(
            test_households,
            residual_after_lfa=np.resize(np.asarray([10.0, np.nan, -5.0, np.inf]), n_households),
            credit_requested=np.resize(np.asarray([8.0, np.nan, -2.0, np.inf]), n_households),
            planned_liquidation=np.resize(np.asarray([2.0, np.nan, -3.0, np.inf]), n_households),
        )

        test_households.populate_post_grant_feasible_plan_from_granted_credit(
            credit_granted=np.resize(np.asarray([5.0, 0.0, -4.0, 0.0]), n_households),
        )

        plan = test_households.post_grant_feasible_plan
        np.testing.assert_allclose(
            plan.credit_granted,
            np.resize(np.asarray([5.0, 0.0, 0.0, 0.0]), n_households),
        )
        np.testing.assert_allclose(
            plan.credit_rationing_gap,
            np.resize(np.asarray([3.0, 0.0, 0.0, 0.0]), n_households),
        )
        np.testing.assert_allclose(
            plan.residual_shortfall_after_granted_credit,
            np.resize(np.asarray([3.0, 0.0, 0.0, 0.0]), n_households),
        )

    def test__post_grant_accessors_return_settled_carrier_values(self, test_households):
        n_households = test_households.ts.current("n_households")
        test_households.post_grant_feasible_plan = households_module.PostGrantFeasiblePlan(
            credit_granted=np.full(n_households, 4.0),
            credit_rationing_gap=np.full(n_households, 2.0),
            planned_liquidation_total=np.full(n_households, 3.0),
            residual_shortfall_after_granted_credit=np.full(n_households, 1.0),
        )

        np.testing.assert_allclose(test_households.current_post_grant_credit_granted(), np.full(n_households, 4.0))
        np.testing.assert_allclose(
            test_households.current_post_grant_credit_rationing_gap(),
            np.full(n_households, 2.0),
        )
        np.testing.assert_allclose(
            test_households.current_post_grant_planned_liquidation_total(),
            np.full(n_households, 3.0),
        )
        np.testing.assert_allclose(
            test_households.current_post_grant_residual_shortfall(),
            np.full(n_households, 1.0),
        )

    def test__post_grant_accessors_raise_without_settled_carrier(self, test_households):
        with pytest.raises(RuntimeError, match="post_grant_feasible_plan"):
            test_households.current_post_grant_residual_shortfall()

    def test__post_grant_accessors_defensively_clip_invalid_values(self, test_households):
        n_households = test_households.ts.current("n_households")
        test_households.post_grant_feasible_plan = households_module.PostGrantFeasiblePlan(
            credit_granted=np.resize(np.asarray([4.0, -2.0, np.nan, np.inf]), n_households),
            credit_rationing_gap=np.zeros(n_households),
            planned_liquidation_total=np.zeros(n_households),
            residual_shortfall_after_granted_credit=np.resize(np.asarray([1.0, -3.0, np.nan, np.inf]), n_households),
        )

        np.testing.assert_allclose(
            test_households.current_post_grant_credit_granted(),
            np.resize(np.asarray([4.0, 0.0, 0.0, 0.0]), n_households),
        )
        np.testing.assert_allclose(
            test_households.current_post_grant_residual_shortfall(),
            np.resize(np.asarray([1.0, 0.0, 0.0, 0.0]), n_households),
        )

    def test__consumption_floor_leaves_plan_unchanged_without_residual_shortfall(self, test_households):
        n_households = test_households.ts.current("n_households")
        test_households.post_grant_feasible_plan = households_module.PostGrantFeasiblePlan(
            credit_granted=np.full(n_households, 4.0),
            credit_rationing_gap=np.full(n_households, 2.0),
            planned_liquidation_total=np.full(n_households, 3.0),
            residual_shortfall_after_granted_credit=np.zeros(n_households),
        )

        test_households.apply_consumption_floor_to_post_grant_plan(
            consumption_before_floor=np.full(n_households, 100.0),
            subsistence_floor=np.full(n_households, 80.0),
        )

        plan = test_households.post_grant_feasible_plan
        np.testing.assert_allclose(plan.consumption_before_floor, np.full(n_households, 100.0))
        np.testing.assert_allclose(plan.residual_shortfall_before_floor, np.zeros(n_households))
        np.testing.assert_allclose(plan.consumption_after_floor, np.full(n_households, 100.0))
        np.testing.assert_allclose(plan.consumption_cut_amount, np.zeros(n_households))
        np.testing.assert_allclose(plan.remaining_subsistence_shortfall, np.zeros(n_households))
        np.testing.assert_array_equal(plan.floor_binding, np.zeros(n_households, dtype=bool))
        np.testing.assert_allclose(plan.credit_granted, np.full(n_households, 4.0))
        np.testing.assert_allclose(plan.credit_rationing_gap, np.full(n_households, 2.0))
        np.testing.assert_allclose(plan.planned_liquidation_total, np.full(n_households, 3.0))
        np.testing.assert_allclose(
            test_households.ts.current("consumption_before_floor"), plan.consumption_before_floor
        )
        np.testing.assert_allclose(
            test_households.ts.current("residual_shortfall_before_floor"),
            plan.residual_shortfall_before_floor,
        )
        np.testing.assert_allclose(test_households.ts.current("consumption_after_floor"), plan.consumption_after_floor)
        np.testing.assert_allclose(test_households.ts.current("consumption_cut_amount"), plan.consumption_cut_amount)
        np.testing.assert_allclose(
            test_households.ts.current("remaining_subsistence_shortfall"),
            plan.remaining_subsistence_shortfall,
        )
        np.testing.assert_array_equal(test_households.ts.current("floor_binding"), plan.floor_binding)

    def test__consumption_floor_reduces_consumption_toward_floor_and_preserves_remaining_shortfall(
        self,
        test_households,
    ):
        n_households = test_households.ts.current("n_households")
        residual = np.resize(np.asarray([5.0, 30.0, 12.0, 0.0]), n_households)
        consumption_before = np.resize(np.asarray([100.0, 100.0, 80.0, 50.0]), n_households)
        subsistence_floor = np.resize(np.asarray([80.0, 85.0, 80.0, 40.0]), n_households)
        test_households.post_grant_feasible_plan = households_module.PostGrantFeasiblePlan(
            credit_granted=np.full(n_households, 4.0),
            credit_rationing_gap=np.full(n_households, 2.0),
            planned_liquidation_total=np.full(n_households, 3.0),
            residual_shortfall_after_granted_credit=residual,
        )

        test_households.apply_consumption_floor_to_post_grant_plan(
            consumption_before_floor=consumption_before,
            subsistence_floor=subsistence_floor,
        )

        plan = test_households.post_grant_feasible_plan
        expected_cut = np.resize(np.asarray([5.0, 15.0, 0.0, 0.0]), n_households)
        expected_consumption_after = consumption_before - expected_cut
        np.testing.assert_allclose(plan.consumption_cut_amount, expected_cut)
        np.testing.assert_allclose(plan.consumption_after_floor, expected_consumption_after)
        np.testing.assert_allclose(
            plan.remaining_subsistence_shortfall,
            np.resize(np.asarray([0.0, 15.0, 12.0, 0.0]), n_households),
        )
        np.testing.assert_array_equal(plan.floor_binding, expected_cut > 0.0)
        np.testing.assert_array_less(plan.consumption_after_floor - 1e-12, consumption_before + 1e-12)
        np.testing.assert_allclose(
            plan.consumption_before_floor,
            plan.consumption_after_floor + plan.consumption_cut_amount,
        )
        np.testing.assert_array_less(subsistence_floor - 1e-12, plan.consumption_after_floor + 1e-12)
        np.testing.assert_allclose(
            test_households.current_remaining_subsistence_shortfall(),
            plan.remaining_subsistence_shortfall,
        )

    def test__consumption_floor_tops_up_below_floor_target_via_subsistence_support(self, test_households):
        n_households = test_households.ts.current("n_households")
        test_households.post_grant_feasible_plan = households_module.PostGrantFeasiblePlan(
            credit_granted=np.zeros(n_households),
            credit_rationing_gap=np.zeros(n_households),
            planned_liquidation_total=np.zeros(n_households),
            residual_shortfall_after_granted_credit=np.zeros(n_households),
        )

        test_households.apply_consumption_floor_to_post_grant_plan(
            consumption_before_floor=np.full(n_households, 50.0),
            subsistence_floor=np.full(n_households, 80.0),
        )

        plan = test_households.post_grant_feasible_plan
        np.testing.assert_allclose(plan.consumption_after_floor, np.full(n_households, 80.0))
        np.testing.assert_allclose(plan.remaining_subsistence_shortfall, np.full(n_households, 30.0))
        np.testing.assert_array_equal(plan.floor_binding, np.ones(n_households, dtype=bool))

    def test__consumption_floor_raises_without_settled_carrier(self, test_households):
        n_households = test_households.ts.current("n_households")

        with pytest.raises(RuntimeError, match="post_grant_feasible_plan"):
            test_households.apply_consumption_floor_to_post_grant_plan(
                consumption_before_floor=np.zeros(n_households),
                subsistence_floor=np.zeros(n_households),
            )

    def test__remaining_subsistence_shortfall_accessor_raises_before_floor_enforcement(self, test_households):
        n_households = test_households.ts.current("n_households")
        test_households.post_grant_feasible_plan = households_module.PostGrantFeasiblePlan(
            credit_granted=np.zeros(n_households),
            credit_rationing_gap=np.zeros(n_households),
            planned_liquidation_total=np.zeros(n_households),
            residual_shortfall_after_granted_credit=np.zeros(n_households),
        )

        with pytest.raises(RuntimeError, match="remaining_subsistence_shortfall"):
            test_households.current_remaining_subsistence_shortfall()

    def test__record_pre_support_payment_suspension_diagnostics_uses_settled_remaining_shortfall(self, test_households):
        n_households = test_households.ts.current("n_households")
        residual = np.resize(np.asarray([0.0, 8.0, 25.0, 4.0]), n_households)
        test_households.post_grant_feasible_plan = households_module.PostGrantFeasiblePlan(
            credit_granted=np.zeros(n_households),
            credit_rationing_gap=np.zeros(n_households),
            planned_liquidation_total=np.zeros(n_households),
            residual_shortfall_after_granted_credit=np.zeros(n_households),
            remaining_subsistence_shortfall=residual.copy(),
        )

        test_households.record_pre_support_payment_suspension_diagnostics(
            scheduled_consumer_payments=np.resize(np.asarray([3.0, 10.0, 10.0, 0.0]), n_households),
            scheduled_mortgage_payments=np.resize(np.asarray([7.0, 2.0, 20.0, 5.0]), n_households),
        )

        np.testing.assert_allclose(
            test_households.ts.current("consumer_payment_suspension_amount"),
            np.resize(np.asarray([0.0, 8.0, 10.0, 0.0]), n_households),
        )
        np.testing.assert_allclose(
            test_households.ts.current("mortgage_payment_suspension_amount"),
            np.resize(np.asarray([0.0, 0.0, 15.0, 4.0]), n_households),
        )
        np.testing.assert_array_equal(
            test_households.ts.current("consumer_payment_suspension_needed"),
            np.resize(np.asarray([False, True, True, False]), n_households),
        )
        np.testing.assert_array_equal(
            test_households.ts.current("mortgage_payment_suspension_needed"),
            np.resize(np.asarray([False, False, True, True]), n_households),
        )

    def test__record_stage6_consumer_distress_state_tracks_misses_and_ficp(self, test_households):
        n_households = test_households.ts.current("n_households")
        scheduled = np.resize(np.asarray([10.0, 10.0, 10.0, 10.0]), n_households)
        actual = np.resize(np.asarray([10.0, 5.0, 10.0, 10.0]), n_households)
        test_households.ts.override_current(
            "consumer_payment_suspension_amount",
            scheduled.copy(),
        )

        test_households.record_stage6_consumer_distress_state(
            scheduled_consumer_payments=scheduled,
            actual_consumer_payments=actual,
            unpaid_consumer_payments=scheduled - actual,
            time_unit=3,
            consumer_contractual_principal=np.zeros(n_households),
            consumer_principal_arrears=np.zeros(n_households),
            consumer_interest_arrears=np.zeros(n_households),
        )

        np.testing.assert_array_equal(
            test_households.ts.current("consumer_distress_state"),
            np.resize(np.asarray([CURRENT, DELINQUENT, CURRENT, CURRENT]), n_households),
        )
        np.testing.assert_array_equal(
            test_households.ts.current("missed_payment_count_consumer"),
            np.resize(np.asarray([0, 1, 0, 0]), n_households),
        )

        actual = np.resize(np.asarray([10.0, 9.0, 10.0, 10.0]), n_households)
        test_households.record_stage6_consumer_distress_state(
            scheduled_consumer_payments=scheduled,
            actual_consumer_payments=actual,
            unpaid_consumer_payments=scheduled - actual,
            time_unit=3,
            consumer_contractual_principal=np.zeros(n_households),
            consumer_principal_arrears=np.zeros(n_households),
            consumer_interest_arrears=np.zeros(n_households),
        )

        np.testing.assert_array_equal(
            test_households.ts.current("consumer_distress_state"),
            np.resize(np.asarray([CURRENT, FICP, CURRENT, CURRENT]), n_households),
        )
        np.testing.assert_array_equal(
            test_households.ts.current("ficp_state"),
            np.resize(np.asarray([False, True, False, False]), n_households),
        )
        np.testing.assert_array_equal(
            test_households.ts.current("ficp_exclusion_remaining_periods"),
            np.resize(np.asarray([0, 20, 0, 0]), n_households),
        )

    def test__record_stage6_ficp_episode_emits_one_horizon_event(self, test_households):
        n_households = test_households.ts.current("n_households")
        scheduled = np.full(n_households, 10.0)

        test_households.record_stage6_consumer_distress_state(
            scheduled_consumer_payments=scheduled,
            actual_consumer_payments=np.full(n_households, 5.0),
            unpaid_consumer_payments=np.full(n_households, 5.0),
            time_unit=3,
            period=0,
            consumer_contractual_principal=np.full(n_households, 100.0),
        )
        test_households.record_stage6_consumer_distress_state(
            scheduled_consumer_payments=scheduled,
            actual_consumer_payments=scheduled,
            unpaid_consumer_payments=np.zeros(n_households),
            time_unit=3,
            period=1,
            consumer_contractual_principal=np.full(n_households, 95.0),
        )
        trigger_events = test_households.record_stage6_consumer_distress_state(
            scheduled_consumer_payments=scheduled,
            actual_consumer_payments=np.full(n_households, 5.0),
            unpaid_consumer_payments=np.full(n_households, 5.0),
            time_unit=3,
            period=2,
            consumer_contractual_principal=np.full(n_households, 90.0),
            consumer_principal_arrears=np.zeros(n_households),
            consumer_interest_arrears=np.zeros(n_households),
        )
        assert trigger_events == ()
        np.testing.assert_array_equal(test_households.ts.current("ficp_episode_id"), np.ones(n_households))
        np.testing.assert_array_equal(
            test_households.ts.current("ficp_exclusion_remaining_periods"), np.full(n_households, 20)
        )

        test_households.ts.override_current("ficp_exclusion_remaining_periods", np.ones(n_households))
        test_households.ts.override_current("ficp_episode_status", np.ones(n_households))
        events = test_households.record_stage6_consumer_distress_state(
            scheduled_consumer_payments=scheduled,
            actual_consumer_payments=scheduled,
            unpaid_consumer_payments=np.zeros(n_households),
            time_unit=3,
            period=22,
            consumer_contractual_principal=np.full(n_households, 40.0),
            consumer_principal_arrears=np.full(n_households, 2.0),
            consumer_interest_arrears=np.full(n_households, 1.0),
        )
        assert len(events) == n_households
        assert {event.ficp_episode_id for event in events} == {1}
        assert {event.horizon_end_period for event in events} == {22}
        np.testing.assert_array_equal(
            test_households.ts.current("ficp_forgiveness_event"), np.ones(n_households, dtype=bool)
        )
        np.testing.assert_allclose(
            test_households.ts.current("ficp_residual_consumer_balance"), np.full(n_households, 43.0)
        )
        np.testing.assert_array_equal(test_households.ts.current("ficp_episode_status"), np.full(n_households, 2))
        np.testing.assert_array_equal(
            test_households.ts.current("ficp_forgiveness_emitted"), np.ones(n_households, dtype=bool)
        )

        test_households.ts.override_current("ficp_forgiveness_event_stage", np.ones(n_households))
        preserved_events = test_households.record_stage6_consumer_distress_state(
            scheduled_consumer_payments=scheduled,
            actual_consumer_payments=scheduled,
            unpaid_consumer_payments=np.zeros(n_households),
            time_unit=3,
            period=23,
            consumer_contractual_principal=np.full(n_households, 39.0),
            consumer_principal_arrears=np.full(n_households, 2.0),
            consumer_interest_arrears=np.full(n_households, 1.0),
        )
        assert preserved_events == ()
        np.testing.assert_array_equal(
            test_households.ts.current("ficp_forgiveness_event"), np.ones(n_households, dtype=bool)
        )
        np.testing.assert_array_equal(test_households.ts.current("ficp_forgiveness_event_stage"), np.ones(n_households))
        np.testing.assert_allclose(
            test_households.ts.current("ficp_forgiveness_event_residual_contractual_principal"),
            np.full(n_households, 40.0),
        )

        # A partially written processed marker must not clear the event before
        # the explicit completed stage is durable.
        test_households.ts.override_current("ficp_forgiveness_event_processed", np.ones(n_households, dtype=bool))
        test_households.ts.override_current("ficp_forgiveness_processed", np.ones(n_households, dtype=bool))
        partial_marker_events = test_households.record_stage6_consumer_distress_state(
            scheduled_consumer_payments=scheduled,
            actual_consumer_payments=scheduled,
            unpaid_consumer_payments=np.zeros(n_households),
            time_unit=3,
            period=24,
            consumer_contractual_principal=np.full(n_households, 39.0),
            consumer_principal_arrears=np.full(n_households, 2.0),
            consumer_interest_arrears=np.full(n_households, 1.0),
        )
        assert partial_marker_events == ()
        np.testing.assert_array_equal(
            test_households.ts.current("ficp_forgiveness_event"), np.ones(n_households, dtype=bool)
        )
        np.testing.assert_array_equal(test_households.ts.current("ficp_forgiveness_event_stage"), np.ones(n_households))

        # Replaying a horizon-end state cannot emit the completed episode again.
        test_households.ts.override_current("ficp_exclusion_remaining_periods", np.ones(n_households))
        test_households.ts.override_current("ficp_episode_status", np.ones(n_households))
        replay_events = test_households.record_stage6_consumer_distress_state(
            scheduled_consumer_payments=scheduled,
            actual_consumer_payments=scheduled,
            unpaid_consumer_payments=np.zeros(n_households),
            time_unit=3,
            period=24,
            consumer_contractual_principal=np.full(n_households, 38.0),
            consumer_principal_arrears=np.zeros(n_households),
            consumer_interest_arrears=np.zeros(n_households),
        )
        assert replay_events == ()

        for household_id in range(n_households):
            test_households.mark_ficp_forgiveness_processed(household_id, 1)

        # The completed episode retains its emitted marker until a later
        # second miss starts a distinct episode.
        test_households.record_stage6_consumer_distress_state(
            scheduled_consumer_payments=scheduled,
            actual_consumer_payments=np.full(n_households, 5.0),
            unpaid_consumer_payments=np.full(n_households, 5.0),
            time_unit=3,
            period=25,
            consumer_contractual_principal=np.full(n_households, 39.0),
        )
        np.testing.assert_array_equal(
            test_households.ts.current("ficp_forgiveness_event"), np.zeros(n_households, dtype=bool)
        )
        np.testing.assert_array_equal(
            test_households.ts.current("ficp_forgiveness_emitted"), np.ones(n_households, dtype=bool)
        )
        test_households.record_stage6_consumer_distress_state(
            scheduled_consumer_payments=scheduled,
            actual_consumer_payments=np.full(n_households, 5.0),
            unpaid_consumer_payments=np.full(n_households, 5.0),
            time_unit=3,
            period=26,
            consumer_contractual_principal=np.full(n_households, 38.0),
            consumer_principal_arrears=np.zeros(n_households),
            consumer_interest_arrears=np.zeros(n_households),
        )
        np.testing.assert_array_equal(test_households.ts.current("ficp_episode_id"), np.full(n_households, 2))
        np.testing.assert_array_equal(test_households.ts.current("ficp_episode_end_period"), np.zeros(n_households))
        np.testing.assert_array_equal(
            test_households.ts.current("ficp_forgiveness_emitted"), np.zeros(n_households, dtype=bool)
        )

    def test__record_stage6_requires_debt_components_for_active_ficp(self, test_households):
        n_households = test_households.ts.current("n_households")
        test_households.ts.override_current("ficp_exclusion_remaining_periods", np.ones(n_households))

        with pytest.raises(ValueError, match="debt components are required"):
            test_households.record_stage6_consumer_distress_state(
                scheduled_consumer_payments=np.full(n_households, 10.0),
                actual_consumer_payments=np.full(n_households, 10.0),
                unpaid_consumer_payments=np.zeros(n_households),
                time_unit=3,
                period=1,
            )

    def test__record_stage6_rejects_non_integer_period_before_mutation(self, test_households):
        n_households = test_households.ts.current("n_households")

        with pytest.raises(ValueError, match="non-negative integer"):
            test_households.record_stage6_consumer_distress_state(
                scheduled_consumer_payments=np.full(n_households, 10.0),
                actual_consumer_payments=np.full(n_households, 10.0),
                unpaid_consumer_payments=np.zeros(n_households),
                time_unit=3,
                period=1.5,
            )

    def test__early_repayment_capacity_is_separate_from_subsistence_shortfall(self, test_households):
        n_households = test_households.ts.current("n_households")
        test_households.ts.override_current("expected_income", np.full(n_households, 100.0))
        test_households.post_grant_feasible_plan = households_module.PostGrantFeasiblePlan(
            credit_granted=np.zeros(n_households),
            credit_rationing_gap=np.zeros(n_households),
            planned_liquidation_total=np.zeros(n_households),
            residual_shortfall_after_granted_credit=np.zeros(n_households),
            consumption_after_floor=np.full(n_households, 20.0),
            remaining_subsistence_shortfall=np.full(n_households, 7.0),
        )

        test_households.populate_post_grant_early_repayment_capacity(
            mortgage_service=np.full(n_households, 30.0),
            scheduled_consumer_service=np.full(n_households, 10.0),
            eligible_ficp=np.arange(n_households) == 0,
        )

        np.testing.assert_allclose(
            test_households.current_early_consumer_repayment_capacity(),
            np.where(np.arange(n_households) == 0, 40.0, 0.0),
        )
        np.testing.assert_allclose(
            test_households.current_remaining_subsistence_shortfall(),
            np.full(n_households, 7.0),
        )

    @pytest.mark.parametrize(
        ("consumption_before_floor", "subsistence_floor"),
        [
            (np.asarray(100.0), np.zeros(3)),
            (np.zeros(2), np.zeros(3)),
            (np.zeros(3), np.zeros((3, 1))),
        ],
    )
    def test__consumption_floor_rejects_bad_input_shapes(
        self,
        test_households,
        consumption_before_floor,
        subsistence_floor,
    ):
        n_households = test_households.ts.current("n_households")
        test_households.post_grant_feasible_plan = households_module.PostGrantFeasiblePlan(
            credit_granted=np.zeros(n_households),
            credit_rationing_gap=np.zeros(n_households),
            planned_liquidation_total=np.zeros(n_households),
            residual_shortfall_after_granted_credit=np.zeros(n_households),
        )

        with pytest.raises(ValueError, match="must contain exactly one value per household"):
            test_households.apply_consumption_floor_to_post_grant_plan(
                consumption_before_floor=consumption_before_floor,
                subsistence_floor=subsistence_floor,
            )

    def test__consumption_floor_clips_non_finite_inputs_without_aliasing(self, test_households):
        n_households = test_households.ts.current("n_households")
        consumption_before = np.resize(np.asarray([100.0, np.nan, -5.0, np.inf]), n_households)
        subsistence_floor = np.resize(np.asarray([80.0, np.nan, -3.0, np.inf]), n_households)
        test_households.post_grant_feasible_plan = households_module.PostGrantFeasiblePlan(
            credit_granted=np.zeros(n_households),
            credit_rationing_gap=np.zeros(n_households),
            planned_liquidation_total=np.zeros(n_households),
            residual_shortfall_after_granted_credit=np.resize(np.asarray([5.0, np.nan, -2.0, np.inf]), n_households),
        )

        test_households.apply_consumption_floor_to_post_grant_plan(
            consumption_before_floor=consumption_before,
            subsistence_floor=subsistence_floor,
        )
        consumption_before[:] = 99.0
        subsistence_floor[:] = 88.0

        plan = test_households.post_grant_feasible_plan
        np.testing.assert_allclose(
            plan.consumption_before_floor,
            np.resize(np.asarray([100.0, 0.0, 0.0, 0.0]), n_households),
        )
        np.testing.assert_allclose(
            plan.residual_shortfall_before_floor,
            np.resize(np.asarray([5.0, 0.0, 0.0, 0.0]), n_households),
        )
        np.testing.assert_allclose(
            plan.consumption_after_floor,
            np.resize(np.asarray([95.0, 0.0, 0.0, 0.0]), n_households),
        )
        np.testing.assert_allclose(
            plan.consumption_cut_amount,
            np.resize(np.asarray([5.0, 0.0, 0.0, 0.0]), n_households),
        )
        np.testing.assert_allclose(plan.remaining_subsistence_shortfall, np.zeros(n_households))


class TestComputeTargetCreditLiveCreditRequested:
    """Stage 5 (feasibility resolver) Increment 5: compute_target_credit() wiring."""

    def test__legacy_compatibility_target_credit_uses_legacy_formula_in_live_diagnostic(self, test_households):
        n_households = test_households.ts.current("n_households")
        test_households.configure_feasibility_resolver(False)
        target_consumption = np.zeros_like(test_households.ts.current("target_consumption"))
        target_consumption[:, 0] = 200.0
        test_households.ts.override_current("target_consumption", target_consumption)
        test_households.ts.override_current("expected_income", np.full(n_households, 50.0))
        test_households.ts.override_current("rent", np.zeros(n_households))
        test_households.ts.override_current("wealth_financial_assets", np.zeros(n_households))
        legacy = test_households.functions["target_credit"].compute_target_consumption_loans(
            target_consumption=test_households.ts.current("target_consumption"),
            income=test_households.ts.current("expected_income"),
            rent=test_households.ts.current("rent"),
            wealth_in_financial_assets=test_households.ts.current("wealth_financial_assets"),
        )

        test_households.compute_target_credit(current_sales=None)

        np.testing.assert_allclose(test_households.ts.current("target_consumption_loans"), legacy)
        np.testing.assert_allclose(test_households.ts.current("live_credit_requested"), legacy)

    def test__compute_target_credit_flag_on_uses_carrier_sentinel_with_no_parallel_recompute(self, test_households):
        n_households = test_households.ts.current("n_households")
        sentinel = np.full(n_households, 12345.0)
        test_households.configure_feasibility_resolver(True)
        test_households.populate_pre_grant_feasible_plan_from_liquid_asset_drawdown(
            liquidity_shortfall_before_repair=np.zeros(n_households),
            funded_from_liquid_assets=np.zeros(n_households),
            residual_shortfall_after_lfa=np.zeros(n_households),
        )
        test_households.populate_pre_grant_feasible_plan_credit_requested(credit_requested=sentinel)
        # Legacy inputs would produce a very different value if any parallel
        # recomputation path (re-deriving from target_consumption etc.) existed.
        target_consumption = np.full_like(test_households.ts.current("target_consumption"), 999.0)
        test_households.ts.override_current("target_consumption", target_consumption)
        test_households.ts.override_current("expected_income", np.zeros(n_households))
        test_households.ts.override_current("rent", np.zeros(n_households))
        test_households.ts.override_current("wealth_financial_assets", np.zeros(n_households))

        test_households.compute_target_credit(current_sales=None)

        np.testing.assert_allclose(test_households.ts.current("target_consumption_loans"), sentinel)
        np.testing.assert_allclose(test_households.ts.current("live_credit_requested"), sentinel)

    def test__compute_target_credit_active_ficp_gate_preserves_mortgage_demand(self, test_households, monkeypatch):
        n_households = test_households.ts.current("n_households")
        sentinel = np.full(n_households, 12345.0)
        mortgage_sentinel = np.full(n_households, 54321.0)
        ficp_remaining = np.zeros(n_households)
        ficp_remaining[1::2] = 20
        test_households.configure_feasibility_resolver(True)
        test_households.populate_pre_grant_feasible_plan_from_liquid_asset_drawdown(
            liquidity_shortfall_before_repair=np.zeros(n_households),
            funded_from_liquid_assets=np.zeros(n_households),
            residual_shortfall_after_lfa=np.zeros(n_households),
        )
        test_households.populate_pre_grant_feasible_plan_credit_requested(credit_requested=sentinel)
        test_households.ts.override_current("ficp_exclusion_remaining_periods", ficp_remaining)
        monkeypatch.setattr(
            test_households.functions["target_credit"],
            "compute_target_mortgage",
            lambda **_kwargs: mortgage_sentinel,
        )

        test_households.compute_target_credit(current_sales=None)

        expected_consumer_demand = sentinel.copy()
        expected_consumer_demand[1::2] = 0.0
        np.testing.assert_allclose(test_households.ts.current("target_consumption_loans"), expected_consumer_demand)
        np.testing.assert_allclose(
            test_households.ts.current("live_credit_requested"),
            expected_consumer_demand,
        )
        np.testing.assert_allclose(test_households.ts.current("target_mortgage"), mortgage_sentinel)

    def test__compute_target_credit_raises_when_enabled_without_populated_credit_requested(self, test_households):
        n_households = test_households.ts.current("n_households")
        test_households.configure_feasibility_resolver(True)
        test_households.populate_pre_grant_feasible_plan_from_liquid_asset_drawdown(
            liquidity_shortfall_before_repair=np.zeros(n_households),
            funded_from_liquid_assets=np.zeros(n_households),
            residual_shortfall_after_lfa=np.zeros(n_households),
        )

        with pytest.raises(RuntimeError, match="credit_requested has not been populated"):
            test_households.compute_target_credit(current_sales=None)


class TestComputeAndRecordBorrowVsSellChoice:
    """Stage 5 (feasibility resolver) Increment 2: borrow-vs-sell diagnostic."""

    def test__uses_stage4_handoff_and_mean_offered_rate(self, test_households, test_banks):
        n_households = test_households.ts.current("n_households")
        residual = np.resize(np.asarray([10.0, 10.0]), n_households)
        test_households.functions["wealth"].phi_1 = 1.0
        test_households.functions["wealth"].lambda_kappa = 0.5
        test_households.ts.override_current(
            "portfolio_delta_tilde",
            np.resize(np.asarray([0.0, -0.1]), n_households),
        )
        test_households.ts.override_current("portfolio_opening_tfa_scale", np.full(n_households, 100.0))
        test_households.ts.override_current("portfolio_post_return_ifa", np.full(n_households, 25.0))
        test_households.ts.override_current("portfolio_target_illiquid_assets", np.full(n_households, 10.0))
        test_households.ts.override_current("portfolio_illiquid_return_rate", np.full(n_households, 0.02))
        test_banks.ts.override_current("interest_rates_on_household_consumption_loans", np.asarray([0.10, 0.14]))
        stage4_handoff = {
            "delta_tilde": test_households.ts.current("portfolio_delta_tilde"),
            "opening_tfa_scale": test_households.ts.current("portfolio_opening_tfa_scale"),
            "post_return_ifa": test_households.ts.current("portfolio_post_return_ifa"),
            "r_kappa": test_households.ts.current("portfolio_illiquid_return_rate"),
        }

        preferred_margin, preferred_amount = test_households.compute_and_record_borrow_vs_sell_choice(
            residual_shortfall_after_lfa=residual,
            banks=test_banks,
            stage4_handoff=stage4_handoff,
        )

        np.testing.assert_array_equal(
            preferred_margin[:2],
            np.asarray([PREFERRED_MARGIN_BORROW, PREFERRED_MARGIN_SELL]),
        )
        np.testing.assert_allclose(preferred_amount[:2], [10.0, 10.0])
        np.testing.assert_allclose(test_households.ts.current("borrow_vs_sell_spread")[0], -0.02)
        np.testing.assert_allclose(test_households.ts.current("borrow_vs_sell_spread")[1], 0.08)

    def test__replace_current_overrides_latest_diagnostics_without_changing_lengths(self, test_households, test_banks):
        n_households = test_households.ts.current("n_households")
        test_households.functions["wealth"].phi_1 = 1.0
        test_households.functions["wealth"].lambda_kappa = 0.5
        test_households.ts.override_current("portfolio_delta_tilde", np.zeros(n_households))
        test_households.ts.override_current("portfolio_opening_tfa_scale", np.full(n_households, 100.0))
        test_households.ts.override_current("portfolio_post_return_ifa", np.full(n_households, 25.0))
        test_households.ts.override_current("portfolio_illiquid_return_rate", np.full(n_households, 0.02))
        test_banks.ts.override_current("interest_rates_on_household_consumption_loans", np.asarray([0.10, 0.14]))
        stage4_handoff = {
            "delta_tilde": test_households.ts.current("portfolio_delta_tilde"),
            "opening_tfa_scale": test_households.ts.current("portfolio_opening_tfa_scale"),
            "post_return_ifa": test_households.ts.current("portfolio_post_return_ifa"),
            "r_kappa": test_households.ts.current("portfolio_illiquid_return_rate"),
        }

        test_households.compute_and_record_borrow_vs_sell_choice(
            np.full(n_households, 10.0),
            test_banks,
            stage4_handoff,
        )
        before_lengths = {
            key: len(test_households.ts.dicts[key])
            for key in [
                "preferred_margin_after_lfa",
                "preferred_margin_amount",
                "borrow_vs_sell_threshold",
                "borrow_vs_sell_spread",
                "borrow_vs_sell_l_tilde",
                "borrow_vs_sell_comparison_valid_flag",
            ]
        }

        preferred_margin, preferred_amount = test_households.compute_and_record_borrow_vs_sell_choice(
            np.full(n_households, 0.0),
            test_banks,
            stage4_handoff,
            replace_current=True,
        )

        after_lengths = {key: len(test_households.ts.dicts[key]) for key in before_lengths}
        assert after_lengths == before_lengths
        np.testing.assert_allclose(preferred_margin, np.zeros(n_households))
        np.testing.assert_allclose(preferred_amount, np.zeros(n_households))

    def test__does_not_mutate_core_balance_sheet_or_credit_targets(self, test_households, test_banks):
        n_households = test_households.ts.current("n_households")
        test_households.functions["wealth"].phi_1 = 1.0
        test_households.functions["wealth"].lambda_kappa = 0.5
        test_households.ts.override_current("portfolio_delta_tilde", np.full(n_households, 1.0))
        test_households.ts.override_current("portfolio_opening_tfa_scale", np.full(n_households, 100.0))
        test_households.ts.override_current("portfolio_post_return_ifa", np.full(n_households, 25.0))
        test_households.ts.override_current("portfolio_illiquid_return_rate", np.full(n_households, 0.02))
        baseline = {
            key: test_households.ts.current(key).copy()
            for key in [
                "liquid_financial_assets",
                "illiquid_financial_assets",
                "target_consumption_loans",
                "target_mortgage",
                "debt_installments",
            ]
        }
        test_banks.ts.override_current("interest_rates_on_household_consumption_loans", np.asarray([0.10, 0.14]))
        stage4_handoff = {
            "delta_tilde": test_households.ts.current("portfolio_delta_tilde"),
            "opening_tfa_scale": test_households.ts.current("portfolio_opening_tfa_scale"),
            "post_return_ifa": test_households.ts.current("portfolio_post_return_ifa"),
            "r_kappa": test_households.ts.current("portfolio_illiquid_return_rate"),
        }

        test_households.compute_and_record_borrow_vs_sell_choice(
            np.full(n_households, 10.0),
            test_banks,
            stage4_handoff,
        )

        for key, values in baseline.items():
            np.testing.assert_allclose(test_households.ts.current(key), values)

    def test__current_stage4_handoff_adds_positive_surplus_before_calling_stage4_helper(
        self, test_households, monkeypatch
    ):
        n_households = test_households.ts.current("n_households")
        test_households.functions["wealth"] = PaperAssetReturnWealthSetter(
            other_real_assets_depreciation_rate=0.05,
            mu_eq=0.0029,
            mu_bond=0.0081,
            sigma_eq=0.0,
            sigma_bond=0.0,
            rho=0.0,
            equity_weight=0.5,
            draw_scope="country_period",
            uses_portfolio_choice=True,
            target_share_source="scalar",
            default_target_illiquid_share=0.65,
            phi_1=5.0,
            lambda_kappa=0.1,
            fixed_cost_share=0.001,
        )
        test_households.ts.override_current("expected_income", np.full(n_households, 200.0))
        test_households.ts.override_current("liquid_financial_assets", np.full(n_households, 50.0))
        test_households.ts.override_current("illiquid_financial_assets", np.full(n_households, 25.0))
        captured = {}

        def fake_stage4_helper(**kwargs):
            captured.update(kwargs)
            return Stage4HouseholdDiagnostics(
                portfolio_opening_tfa_scale=np.full(n_households, 100.0),
                portfolio_target_tfa_base=np.full(n_households, 175.0),
                portfolio_post_return_lfa=kwargs["post_surplus_lfa"],
                portfolio_post_return_ifa=np.full(n_households, 25.0),
                portfolio_investable_surplus=kwargs["investable_surplus"],
                portfolio_target_illiquid_share=np.full(n_households, 0.65),
                portfolio_target_share_clipped_flag=np.zeros(n_households, dtype=bool),
                rebalancing=PortfolioRebalancingResult(
                    portfolio_participates=np.ones(n_households, dtype=bool),
                    actual_illiquid_share=np.zeros(n_households),
                    target_illiquid_assets=np.zeros(n_households),
                    delta_tilde=np.zeros(n_households),
                    kappa_star_tilde=np.zeros(n_households),
                    kappa_tilde=np.zeros(n_households),
                    desired_illiquid_adjustment=np.zeros(n_households),
                    adjustment_cost=np.zeros(n_households),
                    counterfactual_lfa_flow=np.zeros(n_households),
                    counterfactual_ifa_flow=np.zeros(n_households),
                    inaction_flag=np.zeros(n_households, dtype=bool),
                    upper_bound_flag=np.zeros(n_households, dtype=bool),
                    lower_bound_flag=np.zeros(n_households, dtype=bool),
                    infeasible_interval_flag=np.zeros(n_households, dtype=bool),
                    no_financial_assets_flag=np.zeros(n_households, dtype=bool),
                    portfolio_valid_flag=np.ones(n_households, dtype=bool),
                ),
            )

        monkeypatch.setattr(households_module, "compute_stage4_household_diagnostics", fake_stage4_helper)

        test_households.current_stage4_handoff_for_stage5(
            target_consumption_total=np.full(n_households, 120.0),
            scheduled_debt_service=np.full(n_households, 30.0),
        )

        np.testing.assert_allclose(captured["investable_surplus"], np.full(n_households, 50.0))
        np.testing.assert_allclose(captured["post_surplus_lfa"], np.full(n_households, 50.0))

    def test__current_stage4_handoff_uses_post_return_ifa_before_calling_stage4_helper(
        self, test_households, monkeypatch
    ):
        n_households = test_households.ts.current("n_households")
        test_households.functions["wealth"] = PaperAssetReturnWealthSetter(
            other_real_assets_depreciation_rate=0.05,
            mu_eq=0.0029,
            mu_bond=0.0081,
            sigma_eq=0.0,
            sigma_bond=0.0,
            rho=0.0,
            equity_weight=0.5,
            draw_scope="country_period",
            uses_portfolio_choice=True,
            target_share_source="scalar",
            default_target_illiquid_share=0.65,
            phi_1=5.0,
            lambda_kappa=0.1,
            fixed_cost_share=0.001,
        )
        test_households.ts.override_current("illiquid_financial_assets", np.full(n_households, 25.0))
        test_households.ts.override_current("liquid_financial_assets", np.full(n_households, 50.0))
        test_households.ts.override_current("expected_income", np.full(n_households, 200.0))
        test_households.functions["wealth"].compute_income_from_financial_assets(
            current_wealth_in_other_financial_assets=test_households.ts.current("illiquid_financial_assets"),
        )
        expected_post_return_ifa = (
            test_households.ts.current("illiquid_financial_assets")
            + test_households.current_illiquid_financial_asset_return_amount()
        )
        captured = {}

        def fake_stage4_helper(**kwargs):
            captured.update(kwargs)
            return Stage4HouseholdDiagnostics(
                portfolio_opening_tfa_scale=np.full(n_households, 100.0),
                portfolio_target_tfa_base=np.full(n_households, 175.0),
                portfolio_post_return_lfa=kwargs["post_surplus_lfa"],
                portfolio_post_return_ifa=kwargs["post_return_ifa"],
                portfolio_investable_surplus=kwargs["investable_surplus"],
                portfolio_target_illiquid_share=np.full(n_households, 0.65),
                portfolio_target_share_clipped_flag=np.zeros(n_households, dtype=bool),
                rebalancing=PortfolioRebalancingResult(
                    portfolio_participates=np.ones(n_households, dtype=bool),
                    actual_illiquid_share=np.zeros(n_households),
                    target_illiquid_assets=np.zeros(n_households),
                    delta_tilde=np.zeros(n_households),
                    kappa_star_tilde=np.zeros(n_households),
                    kappa_tilde=np.zeros(n_households),
                    desired_illiquid_adjustment=np.zeros(n_households),
                    adjustment_cost=np.zeros(n_households),
                    counterfactual_lfa_flow=np.zeros(n_households),
                    counterfactual_ifa_flow=np.zeros(n_households),
                    inaction_flag=np.zeros(n_households, dtype=bool),
                    upper_bound_flag=np.zeros(n_households, dtype=bool),
                    lower_bound_flag=np.zeros(n_households, dtype=bool),
                    infeasible_interval_flag=np.zeros(n_households, dtype=bool),
                    no_financial_assets_flag=np.zeros(n_households, dtype=bool),
                    portfolio_valid_flag=np.ones(n_households, dtype=bool),
                ),
            )

        monkeypatch.setattr(households_module, "compute_stage4_household_diagnostics", fake_stage4_helper)

        test_households.current_stage4_handoff_for_stage5(
            target_consumption_total=np.full(n_households, 120.0),
            scheduled_debt_service=np.full(n_households, 30.0),
        )

        np.testing.assert_allclose(captured["post_return_ifa"], expected_post_return_ifa)

    def test__current_stage4_handoff_draws_current_return_when_not_pre_drawn(self, test_households, monkeypatch):
        n_households = test_households.ts.current("n_households")
        test_households.functions["wealth"] = PaperAssetReturnWealthSetter(
            other_real_assets_depreciation_rate=0.05,
            mu_eq=0.0029,
            mu_bond=0.0081,
            sigma_eq=0.0,
            sigma_bond=0.0,
            rho=0.0,
            equity_weight=0.5,
            draw_scope="country_period",
            uses_portfolio_choice=True,
            target_share_source="scalar",
            default_target_illiquid_share=0.65,
            phi_1=5.0,
            lambda_kappa=0.1,
            fixed_cost_share=0.001,
        )
        test_households.ts.override_current("illiquid_financial_assets", np.full(n_households, 25.0))
        test_households.ts.override_current("liquid_financial_assets", np.full(n_households, 50.0))
        test_households.ts.override_current("expected_income", np.full(n_households, 200.0))
        monkeypatch.setattr(test_households.functions["wealth"], "draw_illiquid_return_rate", lambda: 0.2)

        handoff = test_households.current_stage4_handoff_for_stage5(
            target_consumption_total=np.full(n_households, 120.0),
            scheduled_debt_service=np.full(n_households, 30.0),
        )

        np.testing.assert_allclose(handoff["post_return_ifa"], np.full(n_households, 30.0))
        np.testing.assert_allclose(handoff["r_kappa"], np.full(n_households, 0.2))


class TestComputeAndRecordResidualCapacityFallback:
    """Stage 5 (feasibility resolver) Increment 3: shadow residual-capacity fallback."""

    def test__records_shadow_plan_without_touching_balance_sheet_state(self, test_households, test_banks):
        n_households = test_households.ts.current("n_households")
        test_banks.ts.override_current("interest_rates_on_household_consumption_loans", np.asarray([0.0, 0.0]))
        preferred_margin_after_lfa = np.resize(
            np.asarray([PREFERRED_MARGIN_BORROW, PREFERRED_MARGIN_SELL]), n_households
        )
        preferred_margin_amount = np.resize(np.asarray([100.0, 25.0]), n_households)
        income = np.full(n_households, 100.0)
        scheduled_mortgage_payment = np.full(n_households, 0.0)
        current_ifa = np.resize(np.asarray([0.0, 10.0]), n_households)
        baseline = {
            key: test_households.ts.current(key).copy()
            for key in [
                "liquid_financial_assets",
                "illiquid_financial_assets",
                "target_consumption_loans",
                "target_mortgage",
                "debt_installments",
            ]
        }

        test_households.compute_and_record_residual_capacity_fallback(
            preferred_margin_after_lfa=preferred_margin_after_lfa,
            preferred_margin_amount=preferred_margin_amount,
            banks=test_banks,
            income=income,
            scheduled_mortgage_payment=scheduled_mortgage_payment,
            consumer_loan_maturity=10,
            dsti_limit=0.1,
            current_ifa=current_ifa,
        )

        np.testing.assert_allclose(test_households.ts.current("dsti_headroom"), np.full(n_households, 10.0))
        np.testing.assert_allclose(
            test_households.ts.current("borrow_planned"),
            np.resize(np.asarray([100.0, 15.0]), n_households),
        )
        np.testing.assert_allclose(
            test_households.ts.current("liquidation_planned"),
            np.resize(np.asarray([0.0, 10.0]), n_households),
        )
        np.testing.assert_allclose(
            test_households.ts.current("shadow_credit_requested"),
            np.resize(np.asarray([100.0, 15.0]), n_households),
        )
        np.testing.assert_allclose(
            test_households.ts.current("forced_liquidation_amount"),
            np.resize(np.asarray([0.0, 0.0]), n_households),
        )
        np.testing.assert_allclose(
            test_households.ts.current("residual_shortfall_after_caps"),
            np.resize(np.asarray([0.0, 0.0]), n_households),
        )
        np.testing.assert_array_equal(
            test_households.ts.current("dsti_cap_binding"),
            np.resize(np.asarray([False, False]), n_households),
        )

        for key, values in baseline.items():
            np.testing.assert_allclose(test_households.ts.current(key), values)

    def test__replace_current_overrides_latest_diagnostics_without_changing_lengths(self, test_households, test_banks):
        n_households = test_households.ts.current("n_households")
        test_banks.ts.override_current("interest_rates_on_household_consumption_loans", np.asarray([0.0, 0.0]))
        preferred_margin_after_lfa = np.full(n_households, PREFERRED_MARGIN_BORROW)
        preferred_margin_amount = np.full(n_households, 12.0)
        income = np.full(n_households, 100.0)
        scheduled_mortgage_payment = np.full(n_households, 0.0)
        current_ifa = np.full(n_households, 0.0)

        test_households.compute_and_record_residual_capacity_fallback(
            preferred_margin_after_lfa=preferred_margin_after_lfa,
            preferred_margin_amount=preferred_margin_amount,
            banks=test_banks,
            income=income,
            scheduled_mortgage_payment=scheduled_mortgage_payment,
            consumer_loan_maturity=10,
            dsti_limit=0.1,
            current_ifa=current_ifa,
        )
        before_lengths = {
            key: len(test_households.ts.dicts[key])
            for key in [
                "dsti_headroom",
                "dsti_maximum_loan_size",
                "dsti_cap_binding",
                "borrow_planned",
                "liquidation_planned",
                "shadow_credit_requested",
                "forced_liquidation_amount",
                "residual_shortfall_after_caps",
            ]
        }

        test_households.compute_and_record_residual_capacity_fallback(
            preferred_margin_after_lfa=preferred_margin_after_lfa,
            preferred_margin_amount=preferred_margin_amount,
            banks=test_banks,
            income=income,
            scheduled_mortgage_payment=scheduled_mortgage_payment,
            consumer_loan_maturity=10,
            dsti_limit=0.1,
            current_ifa=current_ifa,
            replace_current=True,
        )

        after_lengths = {key: len(test_households.ts.dicts[key]) for key in before_lengths}
        assert after_lengths == before_lengths
        np.testing.assert_allclose(test_households.ts.current("borrow_planned"), np.full(n_households, 12.0))
        np.testing.assert_allclose(
            test_households.ts.current("residual_shortfall_after_caps"), np.full(n_households, 0.0)
        )
