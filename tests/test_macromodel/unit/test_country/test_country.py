from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
import pytest

import macromodel.country.country as country_module
from macro_data.readers.permanent_income_forecast import (
    PermanentIncomeForecast,
    load_permanent_income_forecast_inputs,
)
from macro_data.readers.permanent_income_forecast import (
    forecast_common_permanent_income as pure_forecast_common_permanent_income,
)
from macro_data.readers.permanent_income_mapping import (
    design_matrix_to_forecast_reader_names,
    load_permanent_income_design_matrix,
)
from macromodel.agents.households.func.consumption import CreditAugmentedConsumption
from macromodel.agents.households.func.wealth import PaperAssetReturnWealthSetter
from macromodel.configurations import CountryConfiguration, ExchangeRatesConfiguration
from macromodel.country import Country
from macromodel.exchange_rates import ExchangeRates

PERMANENT_INCOME_DATA_PATH = (
    Path(__file__).resolve().parents[4] / "run_model" / "data" / "raw_data" / "permanent_income"
)


def _load_permanent_income_forecast_inputs_for_test():
    """Load FR permanent-income forecast inputs via the production loader."""
    return load_permanent_income_forecast_inputs(
        PERMANENT_INCOME_DATA_PATH / "FR_table.json",
        PERMANENT_INCOME_DATA_PATH / "FR_cov_hac.json",
        PERMANENT_INCOME_DATA_PATH / "FR_sigma2_u.json",
    )


class TestCountry:
    def test__init(self, datawrapper):
        synthetic_country = datawrapper.synthetic_countries["FRA"]
        country_configuration = CountryConfiguration()

        exchange_rates_config = ExchangeRatesConfiguration()
        exchange_rates_df = datawrapper.exchange_rates
        initial_year = 2014
        country_names = ["FRA"]

        exchange_rates = ExchangeRates.from_data(
            exchange_rates_data=exchange_rates_df,
            exchange_rate_config=exchange_rates_config,
            initial_year=initial_year,
            country_names=country_names,
        )

        emission_factors = np.array(
            [
                datawrapper.emission_factors["coal"],
                datawrapper.emission_factors["gas"],
                datawrapper.emission_factors["oil"],
            ]
        )

        country = Country.from_pickled_country(
            synthetic_country=synthetic_country,
            country_configuration=country_configuration,
            exchange_rates=exchange_rates,
            country_name="FRA",
            all_country_names=["FRA", "ROW"],
            industries=datawrapper.industries,
            initial_year=datawrapper.configuration.year,
            t_max=12,
            time_unit=datawrapper.time_unit,
            running_multiple_countries=False,
            emission_factors_usd=emission_factors,
        )

        assert country is not None

    def test__country(self, test_country):
        assert test_country is not None

    def test__income_belief_fields_not_persisted_by_default(self, datawrapper, tmp_path):
        synthetic_country = datawrapper.synthetic_countries["FRA"]
        country_configuration = CountryConfiguration()

        exchange_rates_config = ExchangeRatesConfiguration()
        exchange_rates_df = datawrapper.exchange_rates
        initial_year = 2014
        country_names = ["FRA"]

        exchange_rates = ExchangeRates.from_data(
            exchange_rates_data=exchange_rates_df,
            exchange_rate_config=exchange_rates_config,
            initial_year=initial_year,
            country_names=country_names,
        )

        emission_factors = np.array(
            [
                datawrapper.emission_factors["coal"],
                datawrapper.emission_factors["gas"],
                datawrapper.emission_factors["oil"],
            ]
        )

        country = Country.from_pickled_country(
            synthetic_country=synthetic_country,
            country_configuration=country_configuration,
            exchange_rates=exchange_rates,
            country_name="FRA",
            all_country_names=["FRA", "ROW"],
            industries=datawrapper.industries,
            initial_year=datawrapper.configuration.year,
            t_max=12,
            time_unit=datawrapper.time_unit,
            running_multiple_countries=False,
            emission_factors_usd=emission_factors,
        )

        households = country.households
        assert "income_belief_priors" not in households.states
        assert "income_belief_mu" not in households.ts.get_keys()
        assert "income_belief_p" not in households.ts.get_keys()
        assert "income_belief_rho" not in households.ts.get_keys()

        h5_path = tmp_path / "default_households_no_belief_ts.h5"
        with h5py.File(h5_path, "w") as h5_file:
            country_group = h5_file.create_group("FRA")
            households.save_to_h5(country_group)
            household_group = country_group["households"]
            assert "income_belief_mu" not in household_group
            assert "income_belief_p" not in household_group
            assert "income_belief_rho" not in household_group

    def test__income_belief_priors_initialized_as_static_state_for_opt_in_rule(self, datawrapper):
        synthetic_country = datawrapper.synthetic_countries["FRA"]
        country_configuration = CountryConfiguration()
        country_configuration.households.functions.consumption.name = "CreditAugmentedConsumption"
        country_configuration.households.functions.consumption.parameters = {
            **country_configuration.households.functions.consumption.parameters,
            "uses_income_belief_learning": True,
        }

        exchange_rates_config = ExchangeRatesConfiguration()
        exchange_rates_df = datawrapper.exchange_rates
        initial_year = 2014
        country_names = ["FRA"]

        exchange_rates = ExchangeRates.from_data(
            exchange_rates_data=exchange_rates_df,
            exchange_rate_config=exchange_rates_config,
            initial_year=initial_year,
            country_names=country_names,
        )

        emission_factors = np.array(
            [
                datawrapper.emission_factors["coal"],
                datawrapper.emission_factors["gas"],
                datawrapper.emission_factors["oil"],
            ]
        )

        country = Country.from_pickled_country(
            synthetic_country=synthetic_country,
            country_configuration=country_configuration,
            exchange_rates=exchange_rates,
            country_name="FRA",
            all_country_names=["FRA", "ROW"],
            industries=datawrapper.industries,
            initial_year=datawrapper.configuration.year,
            t_max=12,
            time_unit=datawrapper.time_unit,
            running_multiple_countries=False,
            emission_factors_usd=emission_factors,
        )

        households = country.households
        n_households = households.ts.current("n_households")
        priors = households.states["income_belief_priors"]

        assert priors["income_belief_mu"].shape == (n_households,)
        assert priors["income_belief_p"].shape == (n_households,)
        assert priors["income_belief_rho"].shape == (n_households,)
        assert priors["sigma2_xi"].shape == (n_households,)
        assert priors["sigma2_v"].shape == (n_households,)
        assert np.all(priors["income_belief_p"] >= 0)
        assert np.allclose(priors["income_belief_rho"], 0.9518878627264576)
        assert "income_belief_mu" not in households.ts.get_keys()
        assert "income_belief_p" not in households.ts.get_keys()
        assert "income_belief_rho" not in households.ts.get_keys()

    def test__frm_coefficients_path_prefers_wealth_function_config_over_data_paths(self, datawrapper, tmp_path):
        # Regression guard: country.py must resolve frm_coefficients_path from the
        # wealth function's own config first, falling back to DataPaths only when
        # the wealth function doesn't set one. DataPaths.default_paths() always
        # populates frm_coefficients_path unconditionally (unlike insee_smic_path/
        # income_belief_priors_path, which are genuinely optional), so an
        # inverted precedence would silently discard the YAML-configured path on
        # every real run.
        import json

        from macro_data.readers.default_readers import DataPaths

        real_frm_path = (
            Path(__file__).resolve().parents[4]
            / "run_model"
            / "data"
            / "raw_data"
            / "portfolio"
            / "FR_portfolio_frm_coefficients.json"
        )
        with open(real_frm_path) as f:
            payload = json.load(f)
        # Mutate one coefficient so the two files are distinguishable after loading.
        payload["participation"]["coefficients"]["constant"]["value"] = -99.0
        decoy_frm_path = tmp_path / "decoy_frm_coefficients.json"
        decoy_frm_path.write_text(json.dumps(payload))

        synthetic_country = datawrapper.synthetic_countries["FRA"]
        country_configuration = CountryConfiguration()
        country_configuration.households.functions.wealth.name = "PaperAssetReturnWealthSetter"
        country_configuration.households.functions.wealth.parameters = {
            "other_real_assets_depreciation_rate": 0.05,
            "mu_eq": 0.0029,
            "mu_bond": 0.0081,
            "sigma_eq": 0.0,
            "sigma_bond": 0.0,
            "rho": 0.0,
            "equity_weight": 0.5,
            "draw_scope": "country_period",
            "uses_portfolio_choice": True,
            "target_share_source": "scalar",
            "default_target_illiquid_share": 0.65,
            "phi_1": 5.0,
            "lambda_kappa": 0.1,
            "fixed_cost_share": 0.001,
            "frm_coefficients_path": str(real_frm_path),
        }

        exchange_rates_config = ExchangeRatesConfiguration()
        exchange_rates = ExchangeRates.from_data(
            exchange_rates_data=datawrapper.exchange_rates,
            exchange_rate_config=exchange_rates_config,
            initial_year=2014,
            country_names=["FRA"],
        )
        emission_factors = np.array(
            [
                datawrapper.emission_factors["coal"],
                datawrapper.emission_factors["gas"],
                datawrapper.emission_factors["oil"],
            ]
        )
        # DataPaths always resolves frm_coefficients_path (unconditionally set in
        # default_paths()); point it at the decoy file so a wrong-precedence bug
        # would load the decoy's mutated coefficient instead of the real one.
        data_paths = DataPaths.default_paths(raw_data_path=tmp_path, icio_years=[2014])
        data_paths.frm_coefficients_path = decoy_frm_path

        country = Country.from_pickled_country(
            synthetic_country=synthetic_country,
            country_configuration=country_configuration,
            exchange_rates=exchange_rates,
            country_name="FRA",
            all_country_names=["FRA", "ROW"],
            industries=datawrapper.industries,
            initial_year=datawrapper.configuration.year,
            t_max=12,
            time_unit=datawrapper.time_unit,
            running_multiple_countries=False,
            emission_factors_usd=emission_factors,
            data_paths=data_paths,
        )

        frm_coefficients = country.households.states["frm_coefficients"]
        assert frm_coefficients.participation_coefficients["constant"] != -99.0

    def test__set_household_target_demand_uses_credit_augmented_consumption(self, test_country, monkeypatch, tmp_path):
        test_country.households.functions["consumption"] = CreditAugmentedConsumption(
            consumption_smoothing_fraction=0.0,
            consumption_smoothing_window=1,
            minimum_consumption_fraction=0.0,
        )
        captured = {}
        original = test_country.households.functions["consumption"].compute_target_consumption
        mortgage_payment = np.full(test_country.households.ts.current("n_households"), 123.0)

        def capture(**kwargs):
            captured.update(kwargs)
            return original(**kwargs)

        monkeypatch.setattr(test_country.households.functions["consumption"], "compute_target_consumption", capture)
        monkeypatch.setattr(
            test_country.credit_market,
            "compute_scheduled_mortgage_payments_by_household",
            lambda: mortgage_payment,
        )
        monkeypatch.setattr(
            test_country.credit_market,
            "compute_scheduled_consumption_loan_payments_by_household",
            lambda: np.zeros(test_country.households.ts.current("n_households")),
        )

        test_country._set_household_target_demand(replace_current=False)

        assert "house_price_index" in captured
        assert "house_price_growth" in captured
        assert "liquid_wealth" in captured
        assert "illiquid_wealth" in captured
        assert "lagged_income" in captured
        assert "lagged_cpi" in captured
        assert "lagged_liquid_wealth" in captured
        assert "lagged_illiquid_wealth" in captured
        assert "lagged_mortgage_debt" in captured
        assert "lagged_consumption_loan_debt" in captured
        assert "lagged_housing_wealth" in captured
        assert "lagged_house_price_index" in captured
        assert "real_borrowing_rate" in captured
        assert "consumer_debt_rate_delta" in captured
        assert "owner_occupied" in captured
        assert "mortgagor" in captured
        assert np.allclose(captured["liquid_wealth"], test_country.households.ts.current("wealth_deposits"))
        assert np.allclose(captured["income"], test_country.households.ts.current("expected_income"))
        assert np.allclose(captured["lagged_income"], test_country.households.ts.prev("expected_income"))
        assert np.allclose(captured["lagged_liquid_wealth"], test_country.households.ts.prev("wealth_deposits"))
        assert np.allclose(
            captured["illiquid_wealth"], test_country.households.ts.current("wealth_other_financial_assets")
        )
        assert np.allclose(
            captured["lagged_illiquid_wealth"],
            test_country.households.ts.prev("wealth_other_financial_assets"),
        )
        assert np.allclose(captured["lagged_mortgage_debt"], test_country.households.ts.prev("mortgage_debt"))
        assert np.allclose(
            captured["lagged_consumption_loan_debt"],
            test_country.households.ts.prev("consumption_loan_debt"),
        )
        assert np.allclose(
            captured["housing_wealth"],
            test_country.households.ts.current("wealth_main_residence")
            + test_country.households.ts.current("wealth_other_properties"),
        )
        assert np.allclose(
            captured["lagged_housing_wealth"],
            test_country.households.ts.prev("wealth_main_residence")
            + test_country.households.ts.prev("wealth_other_properties"),
        )
        assert np.allclose(captured["mortgage_payment"], mortgage_payment)
        assert np.allclose(
            captured["owner_occupied"],
            np.isin(test_country.households.states["Tenure Status of the Main Residence"], [1, 2, 4]),
        )
        assert np.allclose(captured["mortgagor"], test_country.households.ts.current("mortgage_debt") > 0.0)

        diagnostic_len = len(test_country.households.ts.historic("formula_implied_mpc"))
        target_len = len(test_country.households.ts.historic("target_consumption"))
        subsistence_len = len(test_country.economy.ts.historic("subsistence_consumption"))
        test_country._set_household_target_demand(replace_current=True)
        assert len(test_country.households.ts.historic("formula_implied_mpc")) == diagnostic_len
        assert len(test_country.households.ts.historic("target_consumption_permanent_income")) == diagnostic_len
        assert len(test_country.households.ts.historic("target_consumption_real_net_liquid_assets")) == diagnostic_len
        assert len(test_country.households.ts.historic("target_consumption_owner_occupied")) == diagnostic_len
        assert len(test_country.households.ts.historic("target_consumption")) == target_len
        assert len(test_country.economy.ts.historic("subsistence_consumption")) == subsistence_len

        h5_path = tmp_path / "households_target_consumption.h5"
        with h5py.File(h5_path, "w") as h5_file:
            country_group = h5_file.create_group("FRA")
            test_country.households.save_to_h5(country_group)
            household_group = country_group["households"]
            assert "formula_implied_mpc" in household_group
            assert "target_consumption_permanent_income" in household_group
            assert "target_consumption_real_income" in household_group
            assert "target_consumption_lagged_real_income" in household_group
            assert "target_consumption_real_net_liquid_assets" in household_group
            assert "target_consumption_real_lagged_house_price" in household_group
            assert "target_consumption_real_lagged_housing_wealth" in household_group
            assert "target_consumption_interest_rate_cashflow" in household_group
            assert "target_consumption_partial_adjustment_gap" in household_group
            assert "target_consumption_owner_occupied" in household_group
            assert "target_consumption_mortgagor" in household_group
            assert "subsistence_consumption_floor" not in household_group
            assert "subsistence_consumption_support" not in household_group

    def test__select_net_smic_base_uses_insee_annual_table(self, monkeypatch):
        table = pd.Series([1234.0], index=pd.Index([2014], name="year"))
        monkeypatch.setattr(country_module, "load_insee_smic_annual_table", lambda path=None: table)

        assert (
            country_module.Country._select_net_smic_base(initial_year=2014, fallback_net_smic=99.0, country_name="FRA")
            == 1234.0
        )

    def test__select_net_smic_base_forwards_smic_path(self, monkeypatch):
        table = pd.Series([1234.0], index=pd.Index([2014], name="year"))
        captured = {}

        def _capture(path=None):
            captured["path"] = path
            return table

        monkeypatch.setattr(country_module, "load_insee_smic_annual_table", _capture)

        country_module.Country._select_net_smic_base(
            initial_year=2014,
            fallback_net_smic=99.0,
            country_name="FRA",
            smic_path="/custom/SMIC.csv",
        )
        assert captured["path"] == "/custom/SMIC.csv"

    def test__select_net_smic_base_falls_back_only_when_year_missing(self, monkeypatch):
        table = pd.Series([1234.0], index=pd.Index([2015], name="year"))
        monkeypatch.setattr(country_module, "load_insee_smic_annual_table", lambda path=None: table)

        assert (
            country_module.Country._select_net_smic_base(initial_year=2014, fallback_net_smic=99.0, country_name="FRA")
            == 99.0
        )

    def test__select_net_smic_base_skips_insee_table_for_non_fra(self, monkeypatch):
        def _raise(path=None):
            raise AssertionError("INSEE SMIC table should not be loaded for non-FRA countries")

        monkeypatch.setattr(country_module, "load_insee_smic_annual_table", _raise)

        assert (
            country_module.Country._select_net_smic_base(initial_year=2014, fallback_net_smic=99.0, country_name="ROW")
            == 99.0
        )

    def test__select_net_smic_base_falls_back_when_value_is_nan(self, monkeypatch):
        table = pd.Series([np.nan], index=pd.Index([2014], name="year"))
        monkeypatch.setattr(country_module, "load_insee_smic_annual_table", lambda path=None: table)

        assert (
            country_module.Country._select_net_smic_base(initial_year=2014, fallback_net_smic=99.0, country_name="FRA")
            == 99.0
        )

    def test__select_net_smic_base_falls_back_on_load_error(self, monkeypatch):
        def _raise(path=None):
            raise OSError("missing file")

        monkeypatch.setattr(country_module, "load_insee_smic_annual_table", _raise)

        assert (
            country_module.Country._select_net_smic_base(initial_year=2014, fallback_net_smic=99.0, country_name="FRA")
            == 99.0
        )

    def test__subsistence_consumption_updates_with_quarterly_cpi(self):
        consumption_units = np.array([1.0, 1.5, 2.0])

        subsistence_consumption = country_module.Country._compute_subsistence_consumption_from_units(
            net_smic_base=100.0,
            current_cpi=1.1,
            initial_cpi=1.0,
            consumption_units=consumption_units,
            time_unit=12,
        )

        np.testing.assert_allclose(subsistence_consumption, [55.0, 82.5, 110.0])

    def test__subsistence_consumption_scales_with_time_unit(self):
        consumption_units = np.array([1.0])

        subsistence_consumption = country_module.Country._compute_subsistence_consumption_from_units(
            net_smic_base=100.0,
            current_cpi=1.0,
            initial_cpi=1.0,
            consumption_units=consumption_units,
            time_unit=3,
        )

        # Monthly net SMIC of 100 scaled to a quarterly period (time_unit=3 months)
        np.testing.assert_allclose(subsistence_consumption, [0.5 * 100.0 * (3 / 12)])

    def test__prepare_post_credit_feasible_activity_plan_revises_labour_and_tax_previews(
        self, test_country, monkeypatch
    ):
        n_firms = test_country.firms.ts.current("n_firms")
        wage_preview = np.full(n_firms, 2.0)
        corporate_tax_preview = np.full(n_firms, 4.0)
        feasible_y = np.full(n_firms, 5.0)
        target_y = np.full(n_firms, 10.0)
        feasible_labour = np.full(n_firms, 3.0)
        interest_preview = np.full(n_firms, 6.0)
        refreshed_wage_markup = np.full(n_firms, 0.25)
        offered_wage_function = object()
        captured = {}

        test_country.assume_zero_growth = True
        test_country.firms.ts.override_current("target_production", target_y)
        test_country.firms.ts.override_current("wage_tightness_markup", np.zeros(n_firms))
        test_country.firms.ts.override_current("activity_finance_feasible_target_production", feasible_y)
        test_country.firms.ts.override_current("activity_finance_feasible_desired_labour_inputs", feasible_labour)
        monkeypatch.setattr(test_country.firms, "compute_total_wage_obligation", lambda **kwargs: wage_preview)
        monkeypatch.setattr(
            test_country.firms,
            "estimate_corporate_tax_obligation",
            lambda **kwargs: corporate_tax_preview,
        )
        monkeypatch.setattr(test_country.firms, "compute_interest_paid_on_deposits", lambda **kwargs: interest_preview)

        def assert_feasible_labour_is_applied_before_wage_refresh():
            assert np.allclose(test_country.firms.ts.current("desired_labour_inputs"), feasible_labour)
            return refreshed_wage_markup

        monkeypatch.setattr(
            test_country.firms,
            "compute_wages_markup",
            assert_feasible_labour_is_applied_before_wage_refresh,
        )
        monkeypatch.setattr(
            test_country.firms,
            "compute_offered_wage_function",
            lambda **kwargs: offered_wage_function,
        )

        def capture_firm_prepare(**kwargs):
            captured.update(kwargs)

        monkeypatch.setattr(test_country.firms, "prepare_feasible_activity_plan", capture_firm_prepare)

        test_country.prepare_post_credit_feasible_activity_plan()

        assert captured["assume_zero_growth"] is True
        assert np.allclose(captured["wage_obligation_preview"], wage_preview)
        assert np.allclose(captured["production_tax_obligation_preview"], 0.0)
        assert np.allclose(captured["corporate_tax_obligation_preview"], 0.0)
        assert np.allclose(test_country.firms.ts.current("desired_labour_inputs"), feasible_labour)
        assert np.allclose(test_country.firms.ts.current("wage_tightness_markup"), refreshed_wage_markup)
        assert test_country.firms.states["offered_wage_function"] is offered_wage_function
        assert np.allclose(test_country._post_credit_corporate_tax_obligation_preview, corporate_tax_preview * 0.5)
        assert np.allclose(
            captured["loan_interest_obligation_preview"],
            test_country.firms.ts.current("firm_settlement_scheduled_interest_due"),
        )

    def test__prepare_goods_market_clearing_uses_prepared_activity_plan(self, test_country, monkeypatch):
        captured = {}

        def capture_firm_orders(**kwargs):
            captured.update(kwargs)

        monkeypatch.setattr(test_country.firms, "prepare_goods_market_orders", capture_firm_orders)
        monkeypatch.setattr(test_country.households, "prepare_goods_market_clearing", lambda **kwargs: None)
        monkeypatch.setattr(test_country.government_entities, "prepare_goods_market_clearing", lambda **kwargs: None)

        test_country.prepare_goods_market_clearing()

        assert captured["exchange_rate_usd_to_lcu"] == test_country.exchange_rate_usd_to_lcu
        assert np.allclose(captured["previous_good_prices"], test_country.economy.ts.current("good_prices"))
        assert np.allclose(
            captured["expected_inflation"],
            test_country.economy.ts.current("estimated_ppi_inflation")[0],
            equal_nan=True,
        )

    def test__excess_demand_finance_diagnostic_only_appends_diagnostics(self, test_country, monkeypatch):
        n_firms = test_country.firms.ts.current("n_firms")
        wage_preview = np.full(n_firms, 2.0)
        interest_preview = np.full(n_firms, 1.0)
        room = {
            "short_term": np.full(n_firms, 3.0),
            "long_term": np.full(n_firms, 7.0),
            "total": np.full(n_firms, 10.0),
        }
        diagnostic_keys = [
            "excess_demand_finance_cash",
            "excess_demand_borrower_st_credit_room",
            "excess_demand_borrower_lt_credit_room",
            "excess_demand_borrower_total_credit_room",
            "excess_demand_repair_cash_used",
            "excess_demand_residual_repair_credit_need",
            "excess_demand_borrower_max_credit",
            "excess_demand_activity_finance_borrower",
            "excess_demand_finance_potential_output_borrower",
            "excess_demand_potential_capacity_borrower",
            "excess_demand_above_borrower_cap_share",
            "excess_demand_supply_max_credit",
            "excess_demand_activity_finance_supply",
            "excess_demand_finance_potential_output_supply",
            "excess_demand_potential_capacity_supply",
            "excess_demand_above_supply_cap_share",
        ]
        core_ts_keys = [
            "target_production",
            "target_intermediate_inputs",
            "target_capital_inputs",
            "deposits",
            "debt",
            "received_credit",
        ]
        test_country.firms.transactor_seller_states["Real Amount sold"] = np.zeros(n_firms)
        test_country.firms.transactor_seller_states["Real Excess Demand"] = np.zeros(n_firms)
        core_state_before = {key: test_country.firms.ts.current(key).copy() for key in core_ts_keys}
        transactor_before = {
            "Real Amount sold": test_country.firms.transactor_seller_states["Real Amount sold"].copy(),
            "Real Excess Demand": test_country.firms.transactor_seller_states["Real Excess Demand"].copy(),
        }
        diagnostic_lengths_before = {key: len(test_country.firms.ts.dicts[key]) for key in diagnostic_keys}

        monkeypatch.setattr(test_country.firms, "compute_total_wage_obligation", lambda **_kwargs: wage_preview)
        monkeypatch.setattr(test_country.firms, "compute_interest_paid_on_deposits", lambda **_kwargs: interest_preview)
        monkeypatch.setattr(country_module, "compute_firm_borrower_credit_room", lambda **_kwargs: room)

        test_country.append_excess_demand_finance_potential_diagnostics()

        for key in core_ts_keys:
            assert np.allclose(test_country.firms.ts.current(key), core_state_before[key], equal_nan=True)
        for key, value in transactor_before.items():
            assert np.allclose(test_country.firms.transactor_seller_states[key], value, equal_nan=True)
        for key in diagnostic_keys:
            assert len(test_country.firms.ts.dicts[key]) == diagnostic_lengths_before[key] + 1

    def test__current_firm_corporate_tax_preview_prefers_post_credit_feasible_preview(self, test_country, monkeypatch):
        n_firms = test_country.firms.ts.current("n_firms")
        feasible_preview = np.full(n_firms, 7.0)

        monkeypatch.setattr(
            test_country.firms,
            "estimate_corporate_tax_obligation",
            lambda **kwargs: np.full(n_firms, 99.0),
        )
        test_country._post_credit_corporate_tax_obligation_preview = feasible_preview

        assert np.allclose(test_country.current_firm_corporate_tax_obligation_preview(), feasible_preview)

    def test__post_labour_realised_feasible_plan_refreshes_activity_tax_previews(self, test_country, monkeypatch):
        n_firms = test_country.firms.ts.current("n_firms")
        target_production = np.full(n_firms, 10.0)
        realised_feasible_y = np.full(n_firms, 4.0)
        tax_rates = np.full_like(test_country.central_government.states["Taxes Less Subsidies Rates"], 0.25)
        prices = np.full(n_firms, 2.0)
        corporate_tax_preview = np.full(n_firms, 5.0)

        test_country.firms.configuration.parameters.firm_activity_finance_revision_mode = "post_credit_cash_budget"
        test_country.assume_zero_growth = False
        test_country.firms.ts.override_current("target_production", target_production)
        test_country.firms.ts.override_current("price", prices)
        test_country.economy.ts.override_current("estimated_ppi_inflation", [0.5])
        test_country.central_government.states["Taxes Less Subsidies Rates"] = tax_rates
        test_country.firms.ts.override_current(
            "activity_finance_realised_feasible_target_production", realised_feasible_y
        )
        monkeypatch.setattr(
            test_country.firms,
            "estimate_corporate_tax_obligation",
            lambda **kwargs: corporate_tax_preview,
        )

        def revise_against_labour(**_kwargs):
            test_country.firms.ts.activity_finance_realised_feasible_target_production.append(realised_feasible_y)
            test_country.firms.ts.activity_finance_realised_labour_scale.append(np.full(n_firms, 0.4))
            test_country.firms.ts.override_current("target_production", realised_feasible_y)

        monkeypatch.setattr(test_country.firms, "revise_activity_against_realised_labour", revise_against_labour)

        test_country.prepare_post_labour_realised_feasible_activity_plan()

        assert np.allclose(
            test_country._post_credit_production_tax_obligation_preview,
            0.25 * realised_feasible_y * prices * 1.5,
        )
        assert np.allclose(test_country._post_credit_corporate_tax_obligation_preview, corporate_tax_preview * 0.4)

    def test__post_labour_realised_feasible_plan_skips_none_mode_with_neutral_diagnostics(self, test_country):
        n_firms = test_country.firms.ts.current("n_firms")
        target_production = np.full(n_firms, 8.0)

        test_country.firms.configuration.parameters.firm_activity_finance_revision_mode = "none"
        test_country.firms.ts.override_current("target_production", target_production)

        test_country.prepare_post_labour_realised_feasible_activity_plan()

        assert np.allclose(
            test_country.firms.ts.current("activity_finance_realised_feasible_target_production"),
            target_production,
        )
        assert np.allclose(test_country.firms.ts.current("activity_finance_realised_labour_scale"), 1.0)

    def test__post_labour_metrics_revises_after_effective_firm_labour_and_before_production(
        self, test_country, monkeypatch
    ):
        events = []
        n_firms = test_country.firms.ts.current("n_firms")
        n_individuals = len(test_country.individuals.states["Corresponding Firm ID"])
        realised_effective_labour = np.full(n_firms, 4.0)

        monkeypatch.setattr(
            test_country.individuals,
            "compute_labour_inputs",
            lambda: np.ones(n_individuals),
        )
        monkeypatch.setattr(
            test_country.firms,
            "compute_n_employees",
            lambda **_kwargs: np.ones(n_firms),
        )

        def compute_firm_labour(**_kwargs):
            events.append("firm_labour")
            test_country.firms.ts.labour_productivity_factor.append(np.ones(n_firms))
            test_country.firms.ts.labour_productivity.append(np.ones(n_firms))
            test_country.firms.ts.labour_inputs.append(realised_effective_labour)
            test_country.firms.ts.normalised_labour_inputs.append(realised_effective_labour)
            return np.ones(n_firms)

        def prepare_post_labour():
            events.append("post_labour_revision")
            assert np.allclose(test_country.firms.ts.current("labour_inputs"), realised_effective_labour)

        def compute_production():
            events.append("production")
            return np.full(n_firms, 2.0)

        monkeypatch.setattr(test_country.firms, "compute_labour_inputs", compute_firm_labour)
        monkeypatch.setattr(test_country, "prepare_post_labour_realised_feasible_activity_plan", prepare_post_labour)
        monkeypatch.setattr(test_country.firms, "compute_production", compute_production)
        monkeypatch.setattr(
            test_country.firms,
            "set_employee_income",
            lambda **_kwargs: np.zeros(n_individuals),
        )
        monkeypatch.setattr(test_country.firms, "compute_price", lambda **_kwargs: np.ones(n_firms))
        monkeypatch.setattr(test_country, "_set_household_income_expectations", lambda **_kwargs: None)
        monkeypatch.setattr(test_country, "_set_household_target_demand", lambda **_kwargs: None)

        test_country.assume_zero_growth = False
        test_country.update_post_labour_planning_metrics()

        assert events == ["firm_labour", "post_labour_revision", "production"]

    def test__household_finance_metrics_are_available_pre_credit_and_replaced_post_labour(
        self, test_country, monkeypatch
    ):
        n_individuals = len(test_country.individuals.states["Activity Status"])
        n_households = test_country.households.ts.current("n_households")
        n_goods = len(test_country.firms.ts.current("price"))
        target_calls = []
        if "properties" not in test_country.housing_market.states:
            test_country.housing_market.states = {"properties": test_country.housing_market.states}

        monkeypatch.setattr(test_country.central_government, "update_benefits", lambda **kwargs: None)
        monkeypatch.setattr(
            test_country.central_government,
            "distribute_unemployment_benefits_to_individuals",
            lambda **kwargs: np.zeros(n_individuals),
        )
        monkeypatch.setattr(
            test_country.individuals,
            "compute_expected_income",
            lambda **kwargs: np.full(n_individuals, len(target_calls) + 1.0),
        )
        monkeypatch.setattr(
            test_country.households,
            "compute_employee_income",
            lambda **kwargs: np.full(n_households, 2.0),
        )
        monkeypatch.setattr(
            test_country.households,
            "compute_expected_social_transfer_income",
            lambda **kwargs: np.full(n_households, 3.0),
        )
        monkeypatch.setattr(
            test_country.households, "compute_rental_income", lambda **kwargs: np.full(n_households, 4.0)
        )
        monkeypatch.setattr(
            test_country.households,
            "compute_expected_income_from_financial_assets",
            lambda: np.full(n_households, 5.0),
        )

        def compute_target_consumption(**_kwargs):
            target_calls.append(None)
            test_country.households.ts.saving_rates_histogram.append([len(target_calls)])
            return np.full((n_households, n_goods), float(len(target_calls)))

        monkeypatch.setattr(test_country.households, "compute_target_consumption", compute_target_consumption)
        monkeypatch.setattr(
            test_country.households,
            "compute_target_investment",
            lambda **_kwargs: np.full(n_households, float(len(target_calls))),
        )
        monkeypatch.setattr(
            test_country.credit_market,
            "compute_scheduled_mortgage_payments_by_household",
            lambda: np.zeros(n_households),
        )
        monkeypatch.setattr(
            test_country.credit_market,
            "compute_scheduled_consumption_loan_payments_by_household",
            lambda: np.zeros(n_households),
        )

        target_len_before = len(test_country.households.ts.target_consumption)
        histogram_len_before = len(test_country.households.ts.saving_rates_histogram)

        test_country.update_pre_credit_household_finance_metrics()

        assert len(test_country.households.ts.target_consumption) == target_len_before + 1
        assert len(test_country.households.ts.saving_rates_histogram) == histogram_len_before + 1
        assert np.allclose(test_country.households.ts.current("target_consumption"), 1.0)
        assert np.allclose(test_country.households.ts.current("expected_income"), 14.0)

        test_country._set_household_income_expectations(replace_current=True)
        test_country._set_household_target_demand(replace_current=True)

        assert len(test_country.households.ts.target_consumption) == target_len_before + 1
        assert len(test_country.households.ts.saving_rates_histogram) == histogram_len_before + 1
        assert np.allclose(test_country.households.ts.current("target_consumption"), 2.0)
        assert np.allclose(test_country.households.ts.current("target_investment"), 2.0)

    def test__prepare_credit_market_clearing_passes_pro_forma_previews(self, test_country, monkeypatch):
        n_firms = test_country.firms.ts.current("n_firms")
        wage_preview = np.full(n_firms, 4.0)
        tax_preview = np.full(n_firms, 5.0)
        loan_interest_preview = np.full(n_firms, 2.0)
        deposit_interest_preview = np.full(n_firms, 7.0)
        captured = {}

        monkeypatch.setattr(test_country.firms, "compute_total_wage_obligation", lambda **kwargs: wage_preview)
        monkeypatch.setattr(
            test_country,
            "compute_pre_credit_production_tax_obligation_preview",
            lambda: tax_preview,
        )
        monkeypatch.setattr(
            test_country.firms, "compute_interest_paid_on_deposits", lambda **kwargs: deposit_interest_preview
        )
        monkeypatch.setattr(
            test_country.credit_market,
            "compute_scheduled_firm_installments_preview",
            lambda: {
                "scheduled_interest_due": loan_interest_preview,
                "scheduled_principal_due": np.full(n_firms, 3.0),
            },
        )

        def capture_firm_credit(**kwargs):
            captured.update(kwargs)

        monkeypatch.setattr(test_country.firms, "compute_target_credit", capture_firm_credit)
        monkeypatch.setattr(test_country.households, "compute_target_credit", lambda **kwargs: None)
        monkeypatch.setattr(test_country.banks, "set_interest_rates", lambda **kwargs: None)
        test_country.housing_market.states["current_sales"] = None

        test_country.prepare_credit_market_clearing()

        assert np.allclose(captured["wage_obligation_preview"], wage_preview)
        assert np.allclose(captured["production_tax_obligation_preview"], tax_preview)
        assert np.allclose(captured["loan_interest_obligation_preview"], loan_interest_preview)
        assert np.allclose(captured["interest_obligation_preview"], loan_interest_preview + deposit_interest_preview)

    def test_household_income_shock_updates_expected_and_realised_income(self, test_country, monkeypatch):
        n_households = test_country.households.ts.current("n_households")
        n_individuals = test_country.individuals.ts.current("expected_income").shape[0]
        base_income = np.full(n_households, 10.0)
        shock = np.full(n_households, 2.5)

        test_country.stage_household_income_shock(shock)

        monkeypatch.setattr(
            test_country.individuals, "compute_expected_income", lambda **_kwargs: np.zeros(n_individuals)
        )
        monkeypatch.setattr(
            test_country.households, "compute_employee_income", lambda **_kwargs: np.zeros(n_households)
        )
        monkeypatch.setattr(
            test_country.households,
            "compute_expected_social_transfer_income",
            lambda **_kwargs: np.zeros(n_households),
        )
        monkeypatch.setattr(test_country.households, "compute_rental_income", lambda **_kwargs: np.zeros(n_households))
        monkeypatch.setattr(
            test_country.households,
            "compute_expected_income_from_financial_assets",
            lambda: np.zeros(n_households),
        )
        monkeypatch.setattr(test_country.households, "compute_expected_income", lambda: base_income)
        monkeypatch.setattr(test_country.households, "compute_income", lambda: base_income)

        test_country._set_household_income_expectations(replace_current=True)

        np.testing.assert_allclose(test_country.households.ts.current("expected_income"), base_income + shock)
        assert test_country._staged_household_income_shock is not None

        realised_income = test_country._apply_household_income_shock(test_country.households.compute_income())
        np.testing.assert_allclose(realised_income, base_income + shock)

        test_country._clear_household_income_shock()
        assert test_country._staged_household_income_shock is None

    def test__housing_market_ratios_are_recorded_after_clearing(self, test_country, monkeypatch):
        calls = []

        def process_housing_market_clearing(**kwargs):
            calls.append("process")
            test_country.housing_market.states["current_sales"] = pd.DataFrame(
                {
                    "sales_types": ["Sell"],
                    "property_id": [0],
                    "property_value": [100.0],
                    "price_or_rent": [200.0],
                    "seller_id": [0],
                    "buyer_id": [1],
                }
            )

        def compute_observed_fraction_value_price():
            assert calls == ["process"]
            calls.append("value")
            return np.array([2.0, 0.0])

        def compute_observed_fraction_rent_value():
            assert calls == ["process", "value"]
            calls.append("rent")
            return np.array([0.01, 0.0])

        monkeypatch.setattr(
            test_country.housing_market,
            "process_housing_market_clearing",
            process_housing_market_clearing,
        )
        monkeypatch.setattr(
            test_country.housing_market,
            "compute_observed_fraction_value_price",
            compute_observed_fraction_value_price,
        )
        monkeypatch.setattr(
            test_country.housing_market,
            "compute_observed_fraction_rent_value",
            compute_observed_fraction_rent_value,
        )
        monkeypatch.setattr(test_country.households, "process_housing_market_clearing", lambda **kwargs: None)

        test_country.process_housing_market_clearing()

        assert calls == ["process", "value", "rent"]
        np.testing.assert_array_equal(
            test_country.housing_market.ts.current("observed_fraction_value_price"),
            np.array([2.0, 0.0]),
        )

    def test__pre_credit_production_tax_preview_uses_target_plan_not_realised_production(self, test_country):
        n_firms = test_country.firms.ts.current("n_firms")
        target_production = np.full(n_firms, 8.0)
        realised_production = np.full(n_firms, 2.0)
        prices = np.full(n_firms, 3.0)
        tax_rates = np.full_like(test_country.central_government.states["Taxes Less Subsidies Rates"], 0.25)

        test_country.firms.ts.override_current("target_production", target_production)
        test_country.firms.ts.override_current("production", realised_production)
        test_country.firms.ts.override_current("price", prices)
        test_country.central_government.states["Taxes Less Subsidies Rates"] = tax_rates
        test_country.economy.ts.override_current("estimated_ppi_inflation", [0.5])

        preview = test_country.compute_pre_credit_production_tax_obligation_preview()

        assert np.allclose(preview, 0.25 * target_production * prices * 1.5)

    def test__forecast_common_permanent_income_returns_none_without_cached_inputs(self, test_country):
        assert test_country._permanent_income_forecast_inputs is None
        assert test_country._permanent_income_design_matrix is None

        assert test_country.forecast_common_permanent_income() is None

    def test__forecast_common_permanent_income_matches_pure_function_and_is_side_effect_free(self, test_country):
        forecast_inputs = _load_permanent_income_forecast_inputs_for_test()
        design_matrix = design_matrix_to_forecast_reader_names(
            load_permanent_income_design_matrix(PERMANENT_INCOME_DATA_PATH / "FR_design_matrix.csv")
        )

        test_country.start_period = pd.Period("2014Q1", freq="Q")
        test_country._permanent_income_forecast_inputs = forecast_inputs
        test_country._permanent_income_design_matrix = design_matrix

        households_income_before = test_country.households.ts.dicts["income"][-1].copy()
        economy_keys_before = set(test_country.economy.ts.get_keys())
        households_keys_before = set(test_country.households.ts.get_keys())

        result = test_country.forecast_common_permanent_income()

        assert result is not None

        sources = test_country._permanent_income_simulation_sources()
        x_t = country_module.build_permanent_income_forecast_regressors(
            sources=sources,
            design_matrix=design_matrix,
            start_period=test_country.start_period,
            estimation_epoch=design_matrix.index.min(),
        )
        expected = pure_forecast_common_permanent_income(x_t, forecast_inputs)

        assert result.point_forecast == expected.point_forecast
        assert result.forecast_variance == expected.forecast_variance

        # Side-effect free: no mutation of household/economy state or learning fields.
        assert np.array_equal(test_country.households.ts.dicts["income"][-1], households_income_before)
        assert set(test_country.economy.ts.get_keys()) == economy_keys_before
        assert set(test_country.households.ts.get_keys()) == households_keys_before
        assert "income_belief_mu" not in test_country.households.ts.get_keys()
        assert "income_belief_p" not in test_country.households.ts.get_keys()

    def test__forecast_common_permanent_income_loaders_called_at_most_once(self, test_country):
        forecast_inputs = _load_permanent_income_forecast_inputs_for_test()
        design_matrix = design_matrix_to_forecast_reader_names(
            load_permanent_income_design_matrix(PERMANENT_INCOME_DATA_PATH / "FR_design_matrix.csv")
        )

        test_country.start_period = pd.Period("2014Q1", freq="Q")
        test_country._permanent_income_forecast_inputs = forecast_inputs
        test_country._permanent_income_design_matrix = design_matrix

        with (
            patch.object(
                country_module, "load_permanent_income_forecast_inputs", wraps=load_permanent_income_forecast_inputs
            ) as mock_load_inputs,
            patch.object(
                country_module, "load_permanent_income_design_matrix", wraps=load_permanent_income_design_matrix
            ) as mock_load_design_matrix,
        ):
            for _ in range(3):
                assert test_country.forecast_common_permanent_income() is not None

        mock_load_inputs.assert_not_called()
        mock_load_design_matrix.assert_not_called()

    def test__forecast_common_permanent_income_returns_none_on_non_positive_base_income(self, test_country, caplog):
        forecast_inputs = _load_permanent_income_forecast_inputs_for_test()
        design_matrix = design_matrix_to_forecast_reader_names(
            load_permanent_income_design_matrix(PERMANENT_INCOME_DATA_PATH / "FR_design_matrix.csv")
        )

        test_country.start_period = pd.Period("2014Q1", freq="Q")
        test_country._permanent_income_forecast_inputs = forecast_inputs
        test_country._permanent_income_design_matrix = design_matrix

        # Force the base-period (first) real_pc_income history value to zero so
        # rebase_real_pc_income_index raises ValueError ("must be finite and positive").
        first_income = test_country.households.ts.dicts["income"][0]
        test_country.households.ts.dicts["income"][0] = np.zeros_like(first_income)

        with caplog.at_level("WARNING"):
            result = test_country.forecast_common_permanent_income()

        assert result is None
        assert any("permanent-income forecast" in record.message for record in caplog.records)

    def test__forecast_common_permanent_income_passes_estimation_epoch_from_design_matrix(self, test_country):
        """build_permanent_income_forecast_regressors must receive estimation_epoch=design_matrix.index.min().

        This pins the fix for the time_trend anchor bug: the trend must be
        counted from the estimation sample's epoch (e.g. 1980Q1 for FR), not
        from start_period (2014Q1).  The test checks that the country wrapper
        passes the correct epoch by capturing the kwarg via a spy.
        """
        forecast_inputs = _load_permanent_income_forecast_inputs_for_test()
        design_matrix = design_matrix_to_forecast_reader_names(
            load_permanent_income_design_matrix(PERMANENT_INCOME_DATA_PATH / "FR_design_matrix.csv")
        )

        test_country.start_period = pd.Period("2014Q1", freq="Q")
        test_country._permanent_income_forecast_inputs = forecast_inputs
        test_country._permanent_income_design_matrix = design_matrix

        captured_kwargs: list[dict] = []
        original_fn = country_module.build_permanent_income_forecast_regressors

        def spy(*args, **kwargs):
            captured_kwargs.append(kwargs)
            return original_fn(*args, **kwargs)

        with patch.object(country_module, "build_permanent_income_forecast_regressors", side_effect=spy):
            result = test_country.forecast_common_permanent_income()

        assert result is not None
        assert len(captured_kwargs) == 1
        expected_epoch = design_matrix.index.min()
        assert captured_kwargs[0]["estimation_epoch"] == expected_epoch

    def test__common_permanent_income_terms_zero_when_forecast_unavailable(self, test_country):
        # Increment 5: no cached forecast inputs -> (0.0, 0.0), preserving the
        # pre-Increment-5 zero behaviour.
        assert test_country._permanent_income_forecast_inputs is None
        assert test_country._common_permanent_income_terms() == (0.0, 0.0)

    def test__common_permanent_income_terms_returns_point_forecast_directly(self, test_country, monkeypatch):
        # forecast.point_forecast is regressed against `ln_y_p_over_y`, i.e. it is
        # already the log permanent-to-current income ratio. It must be returned
        # as-is, with no further level transformation.
        stub_forecast = PermanentIncomeForecast(point_forecast=0.0315, forecast_variance=0.002)
        monkeypatch.setattr(test_country, "forecast_common_permanent_income", lambda: stub_forecast)

        common_log_ratio, common_forecast_variance = test_country._common_permanent_income_terms()
        assert common_log_ratio == pytest.approx(stub_forecast.point_forecast)
        assert common_forecast_variance == pytest.approx(stub_forecast.forecast_variance)

    def test__set_household_target_demand_wires_income_belief_learning_terms(self, test_country, monkeypatch):
        # Increment 5: with the rule opted in, the country call site computes the
        # learning inputs and passes finite, non-None terms to the consumption
        # rule. The common forecast is unavailable here (no cached inputs), so the
        # terms reduce to the pure individual zeta-based component.
        n_households = test_country.households.ts.current("n_households")
        test_country.households.states["income_belief_priors"] = {
            "income_belief_mu": np.linspace(0.05, 0.15, n_households),
            "income_belief_p": np.linspace(0.2, 0.4, n_households),
            "income_belief_rho": np.full(n_households, 0.9519),
            "sigma2_v": np.ones(n_households),
            "sigma2_xi": np.full(n_households, 3.0),
        }
        test_country.households.functions["consumption"] = CreditAugmentedConsumption(
            consumption_smoothing_fraction=0.0,
            consumption_smoothing_window=1,
            minimum_consumption_fraction=0.0,
            permanent_income_propensity=0.1,
            uncertainty_propensity=0.1,
            uses_income_belief_learning=True,
            income_belief_learning_horizon={"delta": 0.95, "S": 40},
        )

        captured = {}
        original = test_country.households.functions["consumption"].compute_target_consumption

        def capture(**kwargs):
            captured.update(kwargs)
            return original(**kwargs)

        monkeypatch.setattr(test_country.households.functions["consumption"], "compute_target_consumption", capture)
        monkeypatch.setattr(
            test_country.credit_market,
            "compute_scheduled_mortgage_payments_by_household",
            lambda: np.zeros(n_households),
        )
        monkeypatch.setattr(
            test_country.credit_market,
            "compute_scheduled_consumption_loan_payments_by_household",
            lambda: np.zeros(n_households),
        )

        # No cached forecast inputs -> common terms are (0.0, 0.0).
        assert test_country._permanent_income_forecast_inputs is None

        economy_keys_before = set(test_country.economy.ts.get_keys())
        households_income_before = test_country.households.ts.dicts["income"][-1].copy()

        test_country._set_household_target_demand(replace_current=False)

        assert captured["permanent_income_log_ratio"] is not None
        assert captured["uncertainty_delta"] is not None
        assert np.all(np.isfinite(captured["permanent_income_log_ratio"]))
        assert np.all(np.isfinite(captured["uncertainty_delta"]))
        zeta = test_country.households.states["income_belief_zeta"]
        runtime_state = test_country.households.states["income_belief_runtime_state"]
        np.testing.assert_allclose(captured["permanent_income_log_ratio"], zeta * runtime_state["posterior_mean"])
        np.testing.assert_allclose(captured["uncertainty_delta"], (zeta**2) * runtime_state["posterior_variance"])
        # The added call site reads only existing series (side-effect free w.r.t.
        # economy/household income state).
        assert set(test_country.economy.ts.get_keys()) == economy_keys_before
        assert np.array_equal(test_country.households.ts.dicts["income"][-1], households_income_before)

    def test__income_belief_zeta_raises_when_horizon_not_configured(self, test_country):
        # With uses_income_belief_learning=True but no income_belief_learning_horizon,
        # _income_belief_zeta must raise rather than silently defaulting, since zeta
        # has real economic meaning and has no safe default.
        n_households = test_country.households.ts.current("n_households")
        test_country.households.states["income_belief_priors"] = {
            "income_belief_mu": np.zeros(n_households),
            "income_belief_p": np.zeros(n_households),
            "income_belief_rho": np.full(n_households, 0.9519),
            "sigma2_v": np.ones(n_households),
            "sigma2_xi": np.ones(n_households),
        }
        test_country.households.functions["consumption"] = CreditAugmentedConsumption(
            consumption_smoothing_fraction=0.0,
            consumption_smoothing_window=1,
            minimum_consumption_fraction=0.0,
            uses_income_belief_learning=True,
            income_belief_learning_horizon=None,
        )
        with pytest.raises(ValueError, match="income_belief_learning_horizon"):
            test_country.households.current_income_belief_learning_inputs()

    def test__current_income_belief_learning_inputs_none_common_terms_treated_as_zero(self, test_country):
        # When the common forecast is unavailable, both common terms default to
        # None -> 0.0. With income_belief_mu=0 and income_belief_p=0 the outputs
        # must be exactly zero arrays, confirming the None->0.0 path produces
        # the correct additive-identity result.
        n_households = test_country.households.ts.current("n_households")
        test_country.households.states["income_belief_priors"] = {
            "income_belief_mu": np.zeros(n_households),
            "income_belief_p": np.zeros(n_households),
            "income_belief_rho": np.full(n_households, 0.9519),
            "sigma2_v": np.ones(n_households),
            "sigma2_xi": np.ones(n_households),
        }
        test_country.households.functions["consumption"] = CreditAugmentedConsumption(
            consumption_smoothing_fraction=0.0,
            consumption_smoothing_window=1,
            minimum_consumption_fraction=0.0,
            uses_income_belief_learning=True,
            income_belief_learning_horizon={"delta": 0.95, "S": 40},
        )
        result = test_country.households.current_income_belief_learning_inputs(
            common_permanent_income_log_ratio=None,
            common_forecast_variance=None,
        )
        np.testing.assert_array_equal(result["permanent_income_log_ratio"], np.zeros(n_households))
        np.testing.assert_array_equal(result["uncertainty_delta"], np.zeros(n_households))

    def test__set_household_target_demand_disabled_is_bit_identical(self, test_country, monkeypatch):
        # Regression: with income-belief learning disabled (default), the new
        # wiring passes None through, and target consumption + all diagnostics
        # are bit-identical to running with the wiring stripped out (None terms).
        test_country.households.functions["consumption"] = CreditAugmentedConsumption(
            consumption_smoothing_fraction=0.0,
            consumption_smoothing_window=1,
            minimum_consumption_fraction=0.0,
        )
        captured = {}
        original = test_country.households.functions["consumption"].compute_target_consumption

        def capture(**kwargs):
            captured.update(kwargs)
            return original(**kwargs)

        monkeypatch.setattr(test_country.households.functions["consumption"], "compute_target_consumption", capture)
        monkeypatch.setattr(
            test_country.credit_market,
            "compute_scheduled_mortgage_payments_by_household",
            lambda: np.zeros(test_country.households.ts.current("n_households")),
        )
        monkeypatch.setattr(
            test_country.credit_market,
            "compute_scheduled_consumption_loan_payments_by_household",
            lambda: np.zeros(test_country.households.ts.current("n_households")),
        )

        test_country._set_household_target_demand(replace_current=False)

        # Disabled rule: both terms stay None (identical to pre-Increment-5).
        assert captured["permanent_income_log_ratio"] is None
        assert captured["uncertainty_delta"] is None
        target_with_wiring = test_country.households.ts.current("target_consumption").copy()
        components_with_wiring = {
            key: value.copy()
            for key, value in test_country.households.functions[
                "consumption"
            ].last_target_consumption_components.items()
        }

        # Recompute target consumption directly with explicit None terms (the
        # behaviour the wiring is supposed to reproduce when disabled).
        target_baseline = original(**{**captured, "permanent_income_log_ratio": None, "uncertainty_delta": None})
        components_baseline = test_country.households.functions["consumption"].last_target_consumption_components

        np.testing.assert_array_equal(target_with_wiring, target_baseline)
        for key, value in components_with_wiring.items():
            np.testing.assert_array_equal(value, components_baseline[key])

    def test__liquidity_shortfall_diagnostic_uses_expected_income_not_realized_income(self, test_country, monkeypatch):
        # Regression (round-2 review finding): at both call sites of
        # _set_household_target_demand, households.ts.current("income") is
        # last period's realized income (Country.update_realised_metrics()
        # only appends a fresh value later in the simulation loop), while
        # target_consumption is built from this period's expected_income.
        # The diagnostic must use expected_income to stay on the same period
        # basis as the consumption plan it is being compared against.
        n_households = test_country.households.ts.current("n_households")
        realized_income = np.full(n_households, 111.0)
        expected_income = np.full(n_households, 222.0)
        test_country.households.ts.override_current("income", realized_income)
        test_country.households.ts.override_current("expected_income", expected_income)

        monkeypatch.setattr(
            test_country.households,
            "compute_target_consumption",
            lambda **_kwargs: np.zeros((n_households, len(test_country.firms.ts.current("price")))),
        )
        monkeypatch.setattr(
            test_country.credit_market,
            "compute_scheduled_mortgage_payments_by_household",
            lambda: np.zeros(n_households),
        )
        monkeypatch.setattr(
            test_country.credit_market,
            "compute_scheduled_consumption_loan_payments_by_household",
            lambda: np.zeros(n_households),
        )
        test_country.banks.ts.override_current(
            "interest_rates_on_household_consumption_loans",
            np.asarray([0.10, 0.14]),
        )
        monkeypatch.setattr(
            test_country.households,
            "current_stage4_handoff_for_stage5",
            lambda **_kwargs: {
                "delta_tilde": np.zeros(n_households),
                "opening_tfa_scale": np.full(n_households, 100.0),
                "post_return_ifa": np.full(n_households, 25.0),
                "r_kappa": np.full(n_households, 0.02),
            },
        )

        test_country._set_household_target_demand(replace_current=False)

        # target_consumption is all-zero (mocked), scheduled_debt_service is
        # all-zero (mocked), so household_saving == income - 0 == income, and
        # the diagnostic must reflect expected_income (222.0), not income (111.0).
        np.testing.assert_allclose(test_country.households.ts.current("household_saving"), np.full(n_households, 222.0))

    def test__liquid_asset_drawdown_diagnostic_uses_liquidity_shortfall_and_current_deposits(
        self,
        test_country,
        monkeypatch,
    ):
        n_households = test_country.households.ts.current("n_households")
        n_industries = len(test_country.firms.ts.current("price"))
        expected_income = np.full(n_households, 100.0)
        target_consumption = np.zeros((n_households, n_industries))
        target_consumption[:, 0] = 300.0
        deposits = np.resize(np.asarray([80.0, 300.0, -5.0]), n_households)
        expected_shortfall = np.full(n_households, 250.0)
        expected_funded = np.minimum(expected_shortfall, np.maximum(deposits, 0.0))
        expected_residual = expected_shortfall - expected_funded

        test_country.households.ts.override_current("expected_income", expected_income)
        test_country.households.ts.override_current("wealth_deposits", deposits)
        pre_call_series = {
            key: test_country.households.ts.current(key).copy()
            for key in [
                "target_consumption",
                "target_consumption_loans",
                "target_mortgage",
                "wealth_deposits",
                "wealth_other_financial_assets",
                "debt_installments",
            ]
            if key in test_country.households.ts.dicts
        }

        monkeypatch.setattr(
            test_country.households,
            "compute_target_consumption",
            lambda **_kwargs: target_consumption,
        )
        monkeypatch.setattr(
            test_country.credit_market,
            "compute_scheduled_mortgage_payments_by_household",
            lambda: np.full(n_households, 50.0),
        )
        monkeypatch.setattr(
            test_country.credit_market,
            "compute_scheduled_consumption_loan_payments_by_household",
            lambda: np.zeros(n_households),
        )

        test_country._set_household_target_demand(replace_current=False)

        np.testing.assert_allclose(
            test_country.households.ts.current("liquidity_shortfall_before_repair"),
            expected_shortfall,
        )
        np.testing.assert_allclose(test_country.households.ts.current("funded_from_liquid_assets"), expected_funded)
        np.testing.assert_allclose(
            test_country.households.ts.current("residual_shortfall_after_lfa"),
            expected_residual,
        )
        np.testing.assert_allclose(
            test_country.households.ts.current("preferred_margin_after_lfa"),
            np.where(expected_residual > 0.0, 1.0, 0.0),
        )
        np.testing.assert_allclose(
            test_country.households.ts.current("preferred_margin_amount"),
            expected_residual,
        )
        np.testing.assert_allclose(test_country.households.ts.current("wealth_deposits"), deposits)
        # Increment 1 is diagnostic-only: it must not touch credit targets,
        # wealth stocks, or debt-service state. target_consumption and
        # target_investment are allowed to update in this planning method.
        for key in [
            "target_consumption_loans",
            "target_mortgage",
            "wealth_deposits",
            "wealth_other_financial_assets",
            "debt_installments",
        ]:
            if key in pre_call_series:
                np.testing.assert_allclose(test_country.households.ts.current(key), pre_call_series[key])

    def test__set_household_target_demand_uses_real_stage4_handoff_for_borrow_vs_sell(self, test_country, monkeypatch):
        n_households = test_country.households.ts.current("n_households")
        n_industries = len(test_country.firms.ts.current("price"))
        test_country.households.functions["wealth"] = PaperAssetReturnWealthSetter(
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
            phi_1=1.0,
            lambda_kappa=0.5,
            fixed_cost_share=0.0,
        )
        monkeypatch.setattr(test_country.households.functions["wealth"], "draw_illiquid_return_rate", lambda: 0.02)
        test_country.households.ts.override_current("expected_income", np.full(n_households, 100.0))
        test_country.households.ts.override_current("wealth_deposits", np.full(n_households, 80.0))
        test_country.households.ts.override_current("wealth_other_financial_assets", np.full(n_households, 25.0))
        target_consumption = np.zeros((n_households, n_industries))
        target_consumption[:, 0] = 300.0

        monkeypatch.setattr(test_country.households, "compute_target_consumption", lambda **_kwargs: target_consumption)
        monkeypatch.setattr(
            test_country.credit_market,
            "compute_scheduled_mortgage_payments_by_household",
            lambda: np.full(n_households, 50.0),
        )
        monkeypatch.setattr(
            test_country.credit_market,
            "compute_scheduled_consumption_loan_payments_by_household",
            lambda: np.zeros(n_households),
        )
        test_country.banks.ts.override_current(
            "interest_rates_on_household_consumption_loans", np.asarray([0.10, 0.14])
        )

        test_country._set_household_target_demand(replace_current=False)

        np.testing.assert_allclose(
            test_country.households.ts.current("borrow_vs_sell_comparison_valid_flag"),
            np.ones(n_households),
        )
        assert np.all(np.isfinite(test_country.households.ts.current("borrow_vs_sell_l_tilde")))
        assert np.all(np.isfinite(test_country.households.ts.current("borrow_vs_sell_threshold")))
        assert np.all(np.isfinite(test_country.households.ts.current("borrow_vs_sell_spread")))
        np.testing.assert_allclose(
            test_country.households.ts.current("preferred_margin_after_lfa"),
            np.ones(n_households),
        )

    def test__set_household_target_demand_records_increment_3_shadow_residual_caps(
        self, test_country, monkeypatch
    ):
        n_households = test_country.households.ts.current("n_households")
        n_industries = len(test_country.firms.ts.current("price"))
        test_country.households.functions["wealth"] = PaperAssetReturnWealthSetter(
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
            phi_1=1.0,
            lambda_kappa=0.5,
            fixed_cost_share=0.0,
        )
        monkeypatch.setattr(test_country.households.functions["wealth"], "draw_illiquid_return_rate", lambda: 0.02)
        test_country.households.ts.override_current("expected_income", np.full(n_households, 100.0))
        test_country.households.ts.override_current("wealth_deposits", np.full(n_households, 80.0))
        test_country.households.ts.override_current("wealth_other_financial_assets", np.full(n_households, 25.0))
        target_consumption = np.zeros((n_households, n_industries))
        target_consumption[:, 0] = 300.0

        monkeypatch.setattr(test_country.households, "compute_target_consumption", lambda **_kwargs: target_consumption)
        monkeypatch.setattr(
            test_country.credit_market,
            "compute_scheduled_mortgage_payments_by_household",
            lambda: np.full(n_households, 50.0),
        )
        monkeypatch.setattr(
            test_country.credit_market,
            "compute_scheduled_consumption_loan_payments_by_household",
            lambda: np.zeros(n_households),
        )
        test_country.banks.ts.override_current(
            "interest_rates_on_household_consumption_loans", np.asarray([0.10, 0.14])
        )

        test_country._set_household_target_demand(replace_current=False)

        np.testing.assert_allclose(test_country.households.ts.current("dsti_headroom"), np.zeros(n_households))
        np.testing.assert_allclose(test_country.households.ts.current("dsti_maximum_loan_size"), np.zeros(n_households))
        np.testing.assert_allclose(test_country.households.ts.current("borrow_planned"), np.zeros(n_households))
        np.testing.assert_allclose(test_country.households.ts.current("liquidation_planned"), np.full(n_households, 25.0))
        np.testing.assert_allclose(test_country.households.ts.current("shadow_credit_requested"), np.zeros(n_households))
        np.testing.assert_allclose(
            test_country.households.ts.current("forced_liquidation_amount"),
            np.full(n_households, 25.0),
        )
        np.testing.assert_allclose(
            test_country.households.ts.current("residual_shortfall_after_caps"),
            np.full(n_households, 145.0),
        )
        np.testing.assert_array_equal(
            test_country.households.ts.current("dsti_cap_binding"),
            np.ones(n_households, dtype=bool),
        )
