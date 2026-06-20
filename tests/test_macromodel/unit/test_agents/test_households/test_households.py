import numpy as np
import pandas as pd
import pytest

import macromodel.agents.households.households as households_module
from macro_data.readers.emission_fraction.emission_fraction_reader import EmissionFractions
from macromodel.agents.households.func.borrow_vs_sell import (
    PREFERRED_MARGIN_BORROW,
    PREFERRED_MARGIN_SELL,
)
from macromodel.agents.households.func.consumption import CreditAugmentedConsumption
from macromodel.agents.households.func.portfolio_diagnostics import Stage4HouseholdDiagnostics
from macromodel.agents.households.func.portfolio_rebalancing import PortfolioRebalancingResult
from macromodel.agents.households.func.wealth import PaperAssetReturnWealthSetter
from macromodel.agents.households.income_belief_learning import compute_zeta


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
            uncertainty_delta=inputs["uncertainty_delta"],
        )
        components = test_households.functions["consumption"].last_target_consumption_components
        np.testing.assert_allclose(
            components["target_consumption_permanent_income_log_ratio"],
            expected_log_ratio,
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

    def test__prepare_goods_market_clearing_reports_subsistence_shortfall_without_altering_demand(
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

        shortfall = test_households.prepare_goods_market_clearing(
            exchange_rate_usd_to_lcu=1.0,
            subsistence_consumption=floor,
        )

        goods_to_buy = test_households.transactor_buyer_states["Initial Goods"]
        expected_goods_budget = target_consumption + target_investment
        np.testing.assert_allclose(test_households.ts.current("target_consumption"), target_consumption)
        np.testing.assert_allclose(goods_to_buy.sum(axis=1), expected_goods_budget.sum(axis=1))
        np.testing.assert_allclose(shortfall, floor - current_consumption_budget)

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
    #         "wealth_deposits",
    #         "wealth_other_financial_assets",
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
    balance-sheet time series it reads from (``wealth_deposits``,
    ``wealth_other_financial_assets``, ``wealth_financial_assets``,
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
            "wealth_deposits",
            "wealth_other_financial_assets",
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

        expected = test_households.ts.current("wealth_other_financial_assets") + test_households.ts.current(
            "wealth_deposits"
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


class TestComputeAndRecordLiquidityShortfall:
    """Stage 5 (feasibility resolver) Increment 0: liquidity-shortfall diagnostic."""

    def test__defaults_to_expected_income_when_no_override_supplied(self, test_households):
        n_households = test_households.ts.current("n_households")
        n_industries = test_households.n_industries
        test_households.ts.override_current("income", np.full(n_households, 111.0))
        test_households.ts.override_current("expected_income", np.full(n_households, 222.0))

        shortfall = test_households.compute_and_record_liquidity_shortfall(
            target_consumption=np.zeros((n_households, n_industries)),
            scheduled_debt_service=np.zeros(n_households),
        )

        # target_consumption and scheduled_debt_service are both zero, so
        # liquidity_shortfall == -income; must reflect expected_income (222),
        # not income (111). See round-2 review finding in households.py's
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
        income_override = np.full(n_households, 999.0)

        shortfall = test_households.compute_and_record_liquidity_shortfall(
            target_consumption=np.zeros((n_households, n_industries)),
            scheduled_debt_service=np.zeros(n_households),
            income_override=income_override,
        )

        np.testing.assert_allclose(shortfall, np.full(n_households, -999.0))


class TestComputeAndRecordLiquidAssetDrawdown:
    """Stage 5 (feasibility resolver) Increment 1: liquid-asset drawdown diagnostic."""

    def test__liquid_asset_drawdown_appends_diagnostics_using_current_wealth_deposits(self, test_households):
        n_households = test_households.ts.current("n_households")
        deposits = np.resize(np.asarray([50.0, 20.0, -5.0]), n_households)
        liquidity_shortfall = np.resize(np.asarray([100.0, 10.0, 30.0]), n_households)
        expected_funded = np.resize(np.asarray([50.0, 10.0, 0.0]), n_households)
        expected_residual = np.resize(np.asarray([50.0, 0.0, 30.0]), n_households)
        test_households.ts.override_current("wealth_deposits", deposits)

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
        test_households.ts.override_current("wealth_deposits", np.full(n_households, 5.0))

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
        test_households.ts.override_current("wealth_deposits", np.full(n_households, 10.0))
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
        np.testing.assert_allclose(test_households.ts.current("borrow_vs_sell_spread")[0], 0.0)
        np.testing.assert_allclose(test_households.ts.current("borrow_vs_sell_spread")[1], 0.10)

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
                "wealth_deposits",
                "wealth_other_financial_assets",
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
        test_households.ts.override_current("wealth_deposits", np.full(n_households, 50.0))
        test_households.ts.override_current("wealth_other_financial_assets", np.full(n_households, 25.0))
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
        np.testing.assert_allclose(captured["post_surplus_lfa"], np.full(n_households, 100.0))

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
        test_households.ts.override_current("wealth_other_financial_assets", np.full(n_households, 25.0))
        test_households.ts.override_current("wealth_deposits", np.full(n_households, 50.0))
        test_households.ts.override_current("expected_income", np.full(n_households, 200.0))
        test_households.functions["wealth"].compute_income_from_financial_assets(
            current_wealth_in_other_financial_assets=test_households.ts.current("wealth_other_financial_assets"),
        )
        expected_post_return_ifa = (
            test_households.ts.current("wealth_other_financial_assets")
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
        test_households.ts.override_current("wealth_other_financial_assets", np.full(n_households, 25.0))
        test_households.ts.override_current("wealth_deposits", np.full(n_households, 50.0))
        test_households.ts.override_current("expected_income", np.full(n_households, 200.0))
        monkeypatch.setattr(test_households.functions["wealth"], "draw_illiquid_return_rate", lambda: 0.2)

        handoff = test_households.current_stage4_handoff_for_stage5(
            target_consumption_total=np.full(n_households, 120.0),
            scheduled_debt_service=np.full(n_households, 30.0),
        )

        np.testing.assert_allclose(handoff["post_return_ifa"], np.full(n_households, 30.0))
        np.testing.assert_allclose(handoff["r_kappa"], np.full(n_households, 0.2))
