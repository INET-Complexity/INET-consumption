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

    def test__prepare_goods_market_clearing_passes_activity_finance_inputs(self, test_country, monkeypatch):
        n_firms = test_country.firms.ts.current("n_firms")
        wage_preview = np.full(n_firms, 2.0)
        tax_preview = np.full(n_firms, 3.0)
        captured = {}

        test_country.assume_zero_growth = True
        monkeypatch.setattr(test_country.firms, "compute_total_wage_obligation", lambda **kwargs: wage_preview)
        monkeypatch.setattr(test_country.firms, "compute_taxes_paid_on_production", lambda **kwargs: tax_preview)

        def capture_firm_prepare(**kwargs):
            captured.update(kwargs)

        monkeypatch.setattr(test_country.firms, "prepare_goods_market_clearing", capture_firm_prepare)
        monkeypatch.setattr(test_country.households, "prepare_goods_market_clearing", lambda **kwargs: None)
        monkeypatch.setattr(test_country.government_entities, "prepare_goods_market_clearing", lambda **kwargs: None)

        test_country.prepare_goods_market_clearing()

        assert captured["assume_zero_growth"] is True
        assert np.allclose(captured["wage_obligation_preview"], wage_preview)
        assert np.allclose(captured["production_tax_obligation_preview"], tax_preview)
