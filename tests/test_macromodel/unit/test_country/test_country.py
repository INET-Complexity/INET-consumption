import numpy as np

from macromodel.configurations import CountryConfiguration, ExchangeRatesConfiguration
from macromodel.country import Country
from macromodel.exchange_rates import ExchangeRates


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
        captured = {}

        test_country.assume_zero_growth = True
        test_country.firms.ts.override_current("target_production", target_y)
        test_country.firms.ts.override_current("activity_finance_feasible_target_production", feasible_y)
        test_country.firms.ts.override_current("activity_finance_feasible_desired_labour_inputs", feasible_labour)
        monkeypatch.setattr(test_country.firms, "compute_total_wage_obligation", lambda **kwargs: wage_preview)
        monkeypatch.setattr(
            test_country.firms,
            "estimate_corporate_tax_obligation",
            lambda **kwargs: corporate_tax_preview,
        )
        monkeypatch.setattr(test_country.firms, "compute_interest_paid_on_deposits", lambda **kwargs: interest_preview)

        def capture_firm_prepare(**kwargs):
            captured.update(kwargs)

        monkeypatch.setattr(test_country.firms, "prepare_feasible_activity_plan", capture_firm_prepare)

        test_country.prepare_post_credit_feasible_activity_plan()

        assert captured["assume_zero_growth"] is True
        assert np.allclose(captured["wage_obligation_preview"], wage_preview)
        assert np.allclose(captured["production_tax_obligation_preview"], 0.0)
        assert np.allclose(captured["corporate_tax_obligation_preview"], 0.0)
        assert np.allclose(test_country.firms.ts.current("desired_labour_inputs"), feasible_labour)
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
