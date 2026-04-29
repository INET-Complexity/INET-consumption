import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from macromodel.configurations.economy_configuration import EconomyConfiguration


class _CapturingInflationForecaster:
    def __init__(self):
        self.calls = []

    def forecast_inflation(self, historic_inflation, exogenous_inflation, current_time, assume_zero_noise):
        self.calls.append({"historic_inflation": historic_inflation.copy(), "current_time": current_time})
        return np.array([np.log(1.02)])


def _record_price_period(economy, prices, sectoral_sales, sectoral_household_consumption=None):
    n_industries = economy.n_industries
    prices = np.asarray(prices, dtype=float)
    sectoral_sales = np.asarray(sectoral_sales, dtype=float)
    if sectoral_household_consumption is None:
        sectoral_household_consumption = np.zeros(n_industries)
    sectoral_household_consumption = np.asarray(sectoral_household_consumption, dtype=float)
    firm_real_amount_bought = np.ones((1, n_industries))
    firm_nominal_amount_spent = prices.reshape(1, n_industries)
    empty_purchases = np.zeros((1, n_industries))

    economy.compute_price_indicators(
        firm_real_amount_bought=firm_real_amount_bought,
        firm_nominal_amount_spent=firm_nominal_amount_spent,
        household_real_amount_bought=empty_purchases,
        household_nominal_amount_spent=empty_purchases,
        government_real_amount_bought=empty_purchases,
        government_nominal_amount_spent=empty_purchases,
        firms_real_amount_bought_as_capital_goods=empty_purchases,
        sectoral_producer_sales=sectoral_sales,
        sectoral_household_consumption=sectoral_household_consumption,
    )
    economy.compute_inflation()


class TestEconomy:
    def test__economy_states(self, test_economy):
        assert test_economy is not None

    def test__consumer_price_source_config_defaults_to_fixed_basket_cpi(self):
        config = EconomyConfiguration()

        assert config.consumer_price_index.source == "fixed_basket_cpi"

    def test__consumer_price_source_config_rejects_invalid_source(self):
        with pytest.raises(ValidationError):
            EconomyConfiguration(consumer_price_index={"source": "legacy_cpi"})

    def test__consumer_price_source_config_rejects_legacy_source_fields(self):
        with pytest.raises(ValidationError):
            EconomyConfiguration(consumer_period_inflation_source="fixed_basket_cpi")

    def test__economy_functions_rejects_legacy_inflation_key(self):
        with pytest.raises(ValidationError):
            EconomyConfiguration(
                functions={
                    "inflation": {
                        "name": "InflationManualForecastingAutoReg",
                        "path_name": "inflation",
                        "parameters": {"lags": 1, "value": 0.0},
                    }
                }
            )

    def test__economy_ts(self, test_economy):
        for ts_key in [
            "ppi",
            "cpi_transaction",
            "cfpi",
            "good_prices",
            "unemployment_rate",
            "participation_rate",
            "vacancy_rate",
            "firm_insolvency_rate",
            "bank_insolvency_rate",
            "household_insolvency_rate",
            "total_growth",
            "cpi_transaction_yoy_change",
            "potential_output",
            "output_gap",
            "ppi_fixed",
            "ppi_fixed_pop_change",
            "ppi_fixed_yoy_change",
            "ppi_chained",
            "ppi_chained_pop_change",
            "ppi_chained_yoy_change",
            "ppi_fixed_weights",
            "ppi_chain_weights",
            "ppi_fixed_base_prices",
            "ppi_chain_base_prices",
            "ppi_chain_link_level",
            "sectoral_producer_sales",
            "cpi_fixed_basket",
            "cpi_fixed_basket_pop_change",
            "cpi_fixed_basket_yoy_change",
            "cpi_chained_basket",
            "cpi_chained_basket_pop_change",
            "cpi_chained_basket_yoy_change",
            "cpi_fixed_basket_weights",
            "cpi_chained_basket_weights",
            "cpi_fixed_basket_base_prices",
            "cpi_chained_basket_base_prices",
            "cpi_chained_basket_link_level",
            "sectoral_household_consumption",
        ]:
            assert ts_key in test_economy.ts.get_keys()
        for legacy_ts_key in [
            "cpi",
            "cpi_inflation",
            "cpi_yoy_inflation",
            "cpi_fixed",
            "cpi_fixed_pop_change",
            "cpi_fixed_yoy_change",
            "cpi_chained",
            "cpi_chained_pop_change",
            "cpi_chained_yoy_change",
            "cpi_fixed_weights",
            "cpi_fixed_base_prices",
            "cpi_chain_weights",
            "cpi_chain_base_prices",
            "cpi_chain_link_level",
        ]:
            assert legacy_ts_key not in test_economy.ts.get_keys()

    def test__initial_ppi_fixed_weights_are_normalized(self, test_economy):
        weights = test_economy.ts.current("ppi_fixed_weights")

        assert weights.shape == (test_economy.n_industries,)
        assert weights.sum() == pytest.approx(1.0)
        assert np.all(weights >= 0.0)

    def test__consumer_price_source_helpers_map_transaction_cpi(self, test_economy):
        test_economy.consumer_price_index_source = "transaction_cpi"
        test_economy.ts.dicts["cpi_transaction"] = [[1.0], [1.1]]
        test_economy.ts.dicts["cpi_transaction_pop_change"] = [[0.01], [0.02]]
        test_economy.ts.dicts["cpi_transaction_yoy_change"] = [[0.03], [0.04]]

        assert test_economy.current_consumer_price_level() == pytest.approx(1.1)
        assert test_economy.initial_consumer_price_level() == pytest.approx(1.0)
        assert test_economy.current_consumer_period_inflation() == pytest.approx(0.02)
        assert test_economy.current_consumer_annual_inflation() == pytest.approx(0.04)

    def test__consumer_price_source_helpers_map_fixed_basket_cpi(self, test_economy):
        test_economy.consumer_price_index_source = "fixed_basket_cpi"
        test_economy.ts.dicts["cpi_fixed_basket"] = [[1.0], [1.2]]
        test_economy.ts.dicts["cpi_fixed_basket_pop_change"] = [[0.03], [0.04]]
        test_economy.ts.dicts["cpi_fixed_basket_yoy_change"] = [[0.05], [0.06]]

        assert test_economy.current_consumer_price_level() == pytest.approx(1.2)
        assert test_economy.initial_consumer_price_level() == pytest.approx(1.0)
        assert test_economy.current_consumer_period_inflation() == pytest.approx(0.04)
        assert test_economy.current_consumer_annual_inflation() == pytest.approx(0.06)

    def test__consumer_price_source_helpers_map_chained_basket_cpi(self, test_economy):
        test_economy.consumer_price_index_source = "chained_basket_cpi"
        test_economy.ts.dicts["cpi_chained_basket"] = [[1.0], [1.3]]
        test_economy.ts.dicts["cpi_chained_basket_pop_change"] = [[0.07], [0.08]]
        test_economy.ts.dicts["cpi_chained_basket_yoy_change"] = [[0.09], [0.10]]

        assert test_economy.current_consumer_price_level() == pytest.approx(1.3)
        assert test_economy.initial_consumer_price_level() == pytest.approx(1.0)
        assert test_economy.current_consumer_period_inflation() == pytest.approx(0.08)
        assert test_economy.current_consumer_annual_inflation() == pytest.approx(0.10)

    def test__initial_cpi_fixed_basket_weights_are_normalized(self, test_economy):
        weights = test_economy.ts.current("cpi_fixed_basket_weights")

        assert weights.shape == (test_economy.n_industries,)
        assert weights.sum() == pytest.approx(1.0)
        assert np.all(weights >= 0.0)

    def test__compute_laspeyres_ppi_fixed(self, test_economy):
        weights = np.zeros(test_economy.n_industries)
        weights[:3] = [0.2, 0.3, 0.5]
        base_prices = np.ones(test_economy.n_industries)
        prices = np.ones(test_economy.n_industries)
        prices[:3] = [2.0, 1.0, 3.0]
        previous_legacy_ppi = test_economy.ts.current("ppi")[0]
        previous_transaction_cpi = test_economy.ts.current("cpi_transaction")[0]

        test_economy.ts.dicts["ppi_fixed_weights"] = [weights]
        test_economy.ts.dicts["ppi_chain_weights"] = [weights]
        test_economy.ts.dicts["ppi_fixed_base_prices"] = [base_prices]
        test_economy.ts.dicts["ppi_chain_base_prices"] = [base_prices]

        _record_price_period(test_economy, prices=prices, sectoral_sales=np.ones(test_economy.n_industries))

        legacy_price_scalar = test_economy.ts.initial("initial_price")[0][0]
        expected_transaction_price_index = prices.mean() / legacy_price_scalar
        assert test_economy.ts.current("ppi_fixed")[0] == pytest.approx(2.2)
        assert test_economy.ts.current("ppi_fixed_pop_change")[0] == pytest.approx(1.2)
        assert test_economy.ts.current("ppi")[0] == pytest.approx(expected_transaction_price_index)
        assert test_economy.ts.current("ppi_inflation")[0] == pytest.approx(
            expected_transaction_price_index / previous_legacy_ppi - 1.0
        )
        assert test_economy.ts.current("cpi_transaction")[0] == pytest.approx(expected_transaction_price_index)
        assert test_economy.ts.current("cpi_transaction_pop_change")[0] == pytest.approx(
            expected_transaction_price_index / previous_transaction_cpi - 1.0
        )

    def test__compute_laspeyres_cpi_fixed_basket(self, test_economy):
        weights = np.zeros(test_economy.n_industries)
        weights[:3] = [0.6, 0.3, 0.1]
        base_prices = np.ones(test_economy.n_industries)
        prices = np.ones(test_economy.n_industries)
        prices[:3] = [2.0, 1.0, 3.0]
        previous_transaction_cpi = test_economy.ts.current("cpi_transaction")[0]

        test_economy.ts.dicts["cpi_fixed_basket_weights"] = [weights]
        test_economy.ts.dicts["cpi_chained_basket_weights"] = [weights]
        test_economy.ts.dicts["cpi_fixed_basket_base_prices"] = [base_prices]
        test_economy.ts.dicts["cpi_chained_basket_base_prices"] = [base_prices]

        _record_price_period(
            test_economy,
            prices=prices,
            sectoral_sales=np.ones(test_economy.n_industries),
            sectoral_household_consumption=weights,
        )

        legacy_price_scalar = test_economy.ts.initial("initial_price")[0][0]
        expected_transaction_price_index = prices.mean() / legacy_price_scalar
        assert test_economy.ts.current("cpi_fixed_basket")[0] == pytest.approx(1.8)
        assert test_economy.ts.current("cpi_fixed_basket_pop_change")[0] == pytest.approx(0.8)
        assert test_economy.ts.current("cpi_transaction")[0] == pytest.approx(expected_transaction_price_index)
        assert test_economy.ts.current("cpi_transaction_pop_change")[0] == pytest.approx(
            expected_transaction_price_index / previous_transaction_cpi - 1.0
        )

    def test__price_relatives_use_neutral_value_for_invalid_prices(self, test_economy):
        relatives = test_economy._price_relatives(
            current_prices=np.array([2.0, np.nan, 4.0, 5.0]),
            base_prices=np.array([1.0, 2.0, 0.0, np.nan]),
        )

        np.testing.assert_allclose(relatives, np.array([2.0, 1.0, 1.0, 1.0]))

    def test__compute_ppi_fixed_yoy_change_uses_quarterly_time_unit(self, test_economy):
        test_economy.time_unit = 3
        test_economy.ts.dicts["ppi_fixed"] = [[1.0], [1.1], [1.2], [1.3], [1.4]]

        assert test_economy._compute_index_yoy_change("ppi_fixed") == pytest.approx(0.4)

    def test__compute_ppi_fixed_yoy_change_uses_monthly_time_unit(self, test_economy):
        test_economy.time_unit = 1
        test_economy.ts.dicts["ppi_fixed"] = [[1.0]] + [[1.0 + 0.01 * i] for i in range(1, 13)]

        assert test_economy._compute_index_yoy_change("ppi_fixed") == pytest.approx(0.12)

    def test__compute_cpi_fixed_basket_yoy_change_uses_quarterly_time_unit(self, test_economy):
        test_economy.time_unit = 3
        test_economy.ts.dicts["cpi_fixed_basket"] = [[1.0], [1.1], [1.2], [1.3], [1.4]]

        assert test_economy._compute_index_yoy_change("cpi_fixed_basket") == pytest.approx(0.4)

    def test__compute_cpi_chained_basket_pop_change(self, test_economy):
        test_economy.ts.dicts["cpi_chained_basket"] = [[1.0], [1.25]]

        assert test_economy._compute_index_pop_change("cpi_chained_basket") == pytest.approx(0.25)

    def test__ppi_chained_weights_update_after_full_model_year(self, test_economy):
        test_economy.time_unit = 3
        n_industries = test_economy.n_industries
        initial_weights = np.zeros(n_industries)
        initial_weights[:2] = [0.5, 0.5]
        base_prices = np.ones(n_industries)
        prices = np.ones(n_industries)

        test_economy.ts.dicts["ppi_fixed_weights"] = [initial_weights]
        test_economy.ts.dicts["ppi_chain_weights"] = [initial_weights]
        test_economy.ts.dicts["ppi_fixed_base_prices"] = [base_prices]
        test_economy.ts.dicts["ppi_chain_base_prices"] = [base_prices]
        test_economy.ts.dicts["ppi_chain_link_level"] = [[1.0]]
        test_economy.ts.dicts["sectoral_producer_sales"] = [np.zeros(n_industries)]

        prior_year_sales = np.zeros(n_industries)
        prior_year_sales[:2] = [1.0, 3.0]
        for _ in range(4):
            _record_price_period(test_economy, prices=prices, sectoral_sales=prior_year_sales)

        np.testing.assert_allclose(test_economy.ts.current("ppi_chain_weights"), initial_weights)

        _record_price_period(test_economy, prices=prices, sectoral_sales=np.ones(n_industries))

        expected_weights = np.zeros(n_industries)
        expected_weights[:2] = [0.25, 0.75]
        np.testing.assert_allclose(test_economy.ts.current("ppi_chain_weights"), expected_weights)
        assert test_economy.ts.current("ppi_chain_link_level")[0] == pytest.approx(test_economy.ts.prev("ppi_chained")[0])
        assert test_economy.ts.current("ppi_chained")[0] == pytest.approx(test_economy.ts.prev("ppi_chained")[0])

    def test__cpi_chained_basket_weights_update_after_full_model_year(self, test_economy):
        test_economy.time_unit = 3
        n_industries = test_economy.n_industries
        initial_weights = np.zeros(n_industries)
        initial_weights[:2] = [0.5, 0.5]
        base_prices = np.ones(n_industries)
        prices = np.ones(n_industries)

        test_economy.ts.dicts["cpi_fixed_basket_weights"] = [initial_weights]
        test_economy.ts.dicts["cpi_chained_basket_weights"] = [initial_weights]
        test_economy.ts.dicts["cpi_fixed_basket_base_prices"] = [base_prices]
        test_economy.ts.dicts["cpi_chained_basket_base_prices"] = [base_prices]
        test_economy.ts.dicts["cpi_chained_basket_link_level"] = [[1.0]]
        test_economy.ts.dicts["sectoral_household_consumption"] = [np.zeros(n_industries)]

        prior_year_consumption = np.zeros(n_industries)
        prior_year_consumption[:2] = [1.0, 3.0]
        for _ in range(4):
            _record_price_period(
                test_economy,
                prices=prices,
                sectoral_sales=np.ones(n_industries),
                sectoral_household_consumption=prior_year_consumption,
            )

        np.testing.assert_allclose(test_economy.ts.current("cpi_chained_basket_weights"), initial_weights)

        _record_price_period(
            test_economy,
            prices=prices,
            sectoral_sales=np.ones(n_industries),
            sectoral_household_consumption=np.ones(n_industries),
        )

        expected_weights = np.zeros(n_industries)
        expected_weights[:2] = [0.25, 0.75]
        np.testing.assert_allclose(test_economy.ts.current("cpi_chained_basket_weights"), expected_weights)
        assert test_economy.ts.current("cpi_chained_basket_link_level")[0] == pytest.approx(test_economy.ts.prev("cpi_chained_basket")[0])
        assert test_economy.ts.current("cpi_chained_basket")[0] == pytest.approx(test_economy.ts.prev("cpi_chained_basket")[0])

    def test__compute_cpi_yoy_inflation(self, test_economy):
        test_economy.ts.dicts["cpi_transaction_pop_change"] = [[0.01], [0.02], [0.03], [0.04]]

        test_economy.compute_cpi_yoy_inflation(exogenous_cpi_inflation_before=np.array([]))

        expected = np.prod([1.01, 1.02, 1.03, 1.04]) - 1.0
        assert test_economy.ts.current("cpi_transaction_yoy_change")[0] == pytest.approx(expected)

    def test__set_estimates_uses_configured_consumer_period_inflation_source(self, test_economy):
        forecaster = _CapturingInflationForecaster()
        test_economy.functions["inflation_forecaster"] = forecaster
        test_economy.consumer_price_index_source = "fixed_basket_cpi"
        test_economy.ts.dicts["cpi_fixed_basket"] = [[1.0], [1.1], [1.2]]
        test_economy.ts.dicts["cpi_fixed_basket_pop_change"] = [[0.10], [0.20], [0.30]]
        test_economy.ts.dicts["cpi_transaction_pop_change"] = [[0.01], [0.02], [0.03]]
        test_economy.ts.dicts["ppi_inflation"] = [[0.01], [0.02], [0.03]]
        test_economy.ts.dicts["ppi"] = [[1.0], [1.1], [1.2]]

        test_economy.set_estimates(
            exogenous_growth=np.array([0.01, 0.01, 0.01]),
            exogenous_inflation=pd.DataFrame(
                {
                    "CPI Inflation": [0.001, 0.002, 0.003],
                    "PPI Inflation": [0.004, 0.005, 0.006],
                }
            ),
            exogenous_hpi_growth=pd.DataFrame(
                {
                    "Real House Price Index Growth": [0.0, 0.0, 0.0],
                    "Nominal House Price Index Growth": [0.0, 0.0, 0.0],
                }
            ),
            forecasting_window=2,
            exogenous_cpi_inflation_during=np.array([0.0, 0.0, 0.0]),
            exogenous_ppi_inflation_during=np.array([0.0, 0.0, 0.0]),
            exogenous_growth_during=np.array([0.0, 0.0, 0.0]),
            assume_zero_noise=True,
        )

        cpi_forecast_call = forecaster.calls[0]
        np.testing.assert_allclose(cpi_forecast_call["historic_inflation"][:2], np.array([0.002, 0.003]))
        np.testing.assert_allclose(cpi_forecast_call["historic_inflation"][2:], np.array([0.10, 0.20, 0.30]))
        assert cpi_forecast_call["current_time"] == 3
        assert test_economy.ts.current("estimated_cpi_inflation")[0] == pytest.approx(0.02)

    def test__compute_output_gap(self, test_economy):
        test_economy.ts.dicts["ppi"] = [[1.0], [1.1]]
        test_economy.ts.dicts["total_output"] = [[100.0], [132.0]]
        test_economy.ts.dicts["potential_output"] = [[100.0]]

        test_economy.compute_output_gap()

        expected_real_output = 132.0 / 1.1
        expected_potential_output = 0.4 * expected_real_output + 0.6 * 100.0
        expected_output_gap = np.log(expected_real_output) - np.log(expected_potential_output)

        assert test_economy.ts.current("real_gross_output")[0] == pytest.approx(expected_real_output)
        assert test_economy.ts.current("potential_output")[0] == pytest.approx(expected_potential_output)
        assert test_economy.ts.current("output_gap")[0] == pytest.approx(expected_output_gap)
