import numpy as np
import pytest


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

    def test__economy_ts(self, test_economy):
        for ts_key in [
            "ppi",
            "cpi",
            "cfpi",
            "good_prices",
            "unemployment_rate",
            "participation_rate",
            "vacancy_rate",
            "firm_insolvency_rate",
            "bank_insolvency_rate",
            "household_insolvency_rate",
            "total_growth",
            "cpi_yoy_inflation",
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
            "cpi_fixed",
            "cpi_fixed_pop_change",
            "cpi_fixed_yoy_change",
            "cpi_chained",
            "cpi_chained_pop_change",
            "cpi_chained_yoy_change",
            "cpi_fixed_weights",
            "cpi_chain_weights",
            "cpi_fixed_base_prices",
            "cpi_chain_base_prices",
            "cpi_chain_link_level",
            "sectoral_household_consumption",
        ]:
            assert ts_key in test_economy.ts.get_keys()

    def test__initial_ppi_fixed_weights_are_normalized(self, test_economy):
        weights = test_economy.ts.current("ppi_fixed_weights")

        assert weights.shape == (test_economy.n_industries,)
        assert weights.sum() == pytest.approx(1.0)
        assert np.all(weights >= 0.0)

    def test__initial_cpi_fixed_weights_are_normalized(self, test_economy):
        weights = test_economy.ts.current("cpi_fixed_weights")

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
        previous_legacy_cpi = test_economy.ts.current("cpi")[0]

        test_economy.ts.dicts["ppi_fixed_weights"] = [weights]
        test_economy.ts.dicts["ppi_chain_weights"] = [weights]
        test_economy.ts.dicts["ppi_fixed_base_prices"] = [base_prices]
        test_economy.ts.dicts["ppi_chain_base_prices"] = [base_prices]

        _record_price_period(test_economy, prices=prices, sectoral_sales=np.ones(test_economy.n_industries))

        legacy_price_scalar = test_economy.ts.initial("initial_price")[0][0]
        expected_legacy_price_index = prices.mean() / legacy_price_scalar
        assert test_economy.ts.current("ppi_fixed")[0] == pytest.approx(2.2)
        assert test_economy.ts.current("ppi_fixed_pop_change")[0] == pytest.approx(1.2)
        assert test_economy.ts.current("ppi")[0] == pytest.approx(expected_legacy_price_index)
        assert test_economy.ts.current("ppi_inflation")[0] == pytest.approx(
            expected_legacy_price_index / previous_legacy_ppi - 1.0
        )
        assert test_economy.ts.current("cpi")[0] == pytest.approx(expected_legacy_price_index)
        assert test_economy.ts.current("cpi_inflation")[0] == pytest.approx(
            expected_legacy_price_index / previous_legacy_cpi - 1.0
        )

    def test__compute_laspeyres_cpi_fixed(self, test_economy):
        weights = np.zeros(test_economy.n_industries)
        weights[:3] = [0.6, 0.3, 0.1]
        base_prices = np.ones(test_economy.n_industries)
        prices = np.ones(test_economy.n_industries)
        prices[:3] = [2.0, 1.0, 3.0]
        previous_legacy_cpi = test_economy.ts.current("cpi")[0]

        test_economy.ts.dicts["cpi_fixed_weights"] = [weights]
        test_economy.ts.dicts["cpi_chain_weights"] = [weights]
        test_economy.ts.dicts["cpi_fixed_base_prices"] = [base_prices]
        test_economy.ts.dicts["cpi_chain_base_prices"] = [base_prices]

        _record_price_period(
            test_economy,
            prices=prices,
            sectoral_sales=np.ones(test_economy.n_industries),
            sectoral_household_consumption=weights,
        )

        legacy_price_scalar = test_economy.ts.initial("initial_price")[0][0]
        expected_legacy_price_index = prices.mean() / legacy_price_scalar
        assert test_economy.ts.current("cpi_fixed")[0] == pytest.approx(1.8)
        assert test_economy.ts.current("cpi_fixed_pop_change")[0] == pytest.approx(0.8)
        assert test_economy.ts.current("cpi")[0] == pytest.approx(expected_legacy_price_index)
        assert test_economy.ts.current("cpi_inflation")[0] == pytest.approx(
            expected_legacy_price_index / previous_legacy_cpi - 1.0
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

    def test__compute_cpi_fixed_yoy_change_uses_quarterly_time_unit(self, test_economy):
        test_economy.time_unit = 3
        test_economy.ts.dicts["cpi_fixed"] = [[1.0], [1.1], [1.2], [1.3], [1.4]]

        assert test_economy._compute_index_yoy_change("cpi_fixed") == pytest.approx(0.4)

    def test__compute_cpi_chained_pop_change(self, test_economy):
        test_economy.ts.dicts["cpi_chained"] = [[1.0], [1.25]]

        assert test_economy._compute_index_pop_change("cpi_chained") == pytest.approx(0.25)

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

    def test__cpi_chained_weights_update_after_full_model_year(self, test_economy):
        test_economy.time_unit = 3
        n_industries = test_economy.n_industries
        initial_weights = np.zeros(n_industries)
        initial_weights[:2] = [0.5, 0.5]
        base_prices = np.ones(n_industries)
        prices = np.ones(n_industries)

        test_economy.ts.dicts["cpi_fixed_weights"] = [initial_weights]
        test_economy.ts.dicts["cpi_chain_weights"] = [initial_weights]
        test_economy.ts.dicts["cpi_fixed_base_prices"] = [base_prices]
        test_economy.ts.dicts["cpi_chain_base_prices"] = [base_prices]
        test_economy.ts.dicts["cpi_chain_link_level"] = [[1.0]]
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

        np.testing.assert_allclose(test_economy.ts.current("cpi_chain_weights"), initial_weights)

        _record_price_period(
            test_economy,
            prices=prices,
            sectoral_sales=np.ones(n_industries),
            sectoral_household_consumption=np.ones(n_industries),
        )

        expected_weights = np.zeros(n_industries)
        expected_weights[:2] = [0.25, 0.75]
        np.testing.assert_allclose(test_economy.ts.current("cpi_chain_weights"), expected_weights)
        assert test_economy.ts.current("cpi_chain_link_level")[0] == pytest.approx(test_economy.ts.prev("cpi_chained")[0])
        assert test_economy.ts.current("cpi_chained")[0] == pytest.approx(test_economy.ts.prev("cpi_chained")[0])

    def test__compute_cpi_yoy_inflation(self, test_economy):
        test_economy.ts.dicts["cpi_inflation"] = [[0.01], [0.02], [0.03], [0.04]]

        test_economy.compute_cpi_yoy_inflation(exogenous_cpi_inflation_before=np.array([]))

        expected = np.prod([1.01, 1.02, 1.03, 1.04]) - 1.0
        assert test_economy.ts.current("cpi_yoy_inflation")[0] == pytest.approx(expected)

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
