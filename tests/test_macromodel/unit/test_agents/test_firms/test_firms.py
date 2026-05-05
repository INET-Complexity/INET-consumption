import numpy as np


class TestFirms:
    def test__firms_states(self, test_firms):
        assert test_firms is not None
        for state in [
            "Industry",
            "Corresponding Bank ID",
            "Employments",
            "is_insolvent",
            "Excess Demand",
            "forced_productivity_investment",
        ]:
            assert state in test_firms.states.keys()

    def test__firms_ts(self, test_firms):
        for ts_key in [
            "n_firms",
            "n_firms_by_industry",
            "number_of_employees",
            "production",
            "target_production",
            "price",
            "price_in_usd",
            "profits",
            "taxes_paid_on_production",
            "corporate_taxes_paid",
            "equity",
            "estimated_demand",
            "demand",
            "unconstrained_target_intermediate_inputs",
            "unconstrained_target_intermediate_inputs_costs",
            "unconstrained_target_capital_inputs",
            "unconstrained_target_capital_inputs_costs",
            "target_intermediate_inputs",
            "target_capital_inputs",
            "inventory",
            "intermediate_inputs_stock",
            "intermediate_inputs_stock_value",
            "used_intermediate_inputs",
            "used_intermediate_inputs_costs",
            "capital_inputs_stock",
            "capital_inputs_stock_value",
            "used_capital_inputs",
            "used_capital_inputs_costs",
            "real_amount_bought_as_intermediate_inputs",
            "real_amount_bought_as_capital_goods",
            "total_sales",
            "target_short_term_credit",
            "target_long_term_credit",
            "received_short_term_credit",
            "received_long_term_credit",
            "received_credit",
            "short_term_loan_debt",
            "long_term_loan_debt",
            "debt",
            "debt_installments",
            "interest_paid_on_deposits",
            "interest_paid_on_loans",
            "interest_paid",
            "deposits",
            "estimated_growth_by_firm",
            "labour_inputs",
            "desired_labour_inputs",
            "labour_costs",
            "real_executed_productivity_investment",
            # "real_amount_bought_as_capital_inputs",
        ]:
            assert ts_key in test_firms.ts.get_keys()

    def test__compute_labour_inputs(self, test_firms):
        assert np.allclose(
            test_firms.compute_labour_inputs(
                corresponding_firm=np.arange(18),
                current_labour_inputs=np.full(18, 1.0),
            ),
            np.full(18, 1.0),
        )

    def test__compute_n_employees(self, test_firms):
        assert np.allclose(
            test_firms.compute_n_employees(corresponding_firm=np.arange(18)),
            np.full(18, 1),
        )

    def test__update_total_wages_paid(self, test_firms):
        test_firms.update_total_wages_paid(
            corresponding_firm=np.arange(18),
            individual_wages=np.full(18, 2.0),
            income_taxes=0.2,
            employee_social_insurance_tax=0.05,
            employer_social_insurance_tax=0.03,
            cpi=1.0,
        )

    def test__compute_inventory(self, test_firms):
        test_firms.ts.real_amount_sold.append(np.full(18, 0.5))
        depreciation_rates = np.array(test_firms.depreciation_rates[test_firms.states["Industry"]])
        a = (1 - depreciation_rates) * (test_firms.ts.current("inventory") + test_firms.ts.current("production"))
        b = test_firms.compute_inventory()
        assert np.allclose(
            a - b,
            np.full(18, 0.5),
        )

    def test__compute_profits(self, test_firms):
        test_firms.ts.used_intermediate_inputs_costs.append(np.full(18, 10.0))
        test_firms.ts.used_capital_inputs_costs.append(np.full(18, 10.0))
        test_firms.ts.taxes_paid_on_production.append(np.full(18, 1.0))
        test_firms.ts.interest_paid.append(np.full(18, 2.0))
        test_firms.ts["interest_received"] = np.full(18, 3.0)
        test_firms.ts.total_wage.append(np.full(18, 1.0))
        assert np.allclose(
            test_firms.compute_profits() - test_firms.ts.current("production") * test_firms.ts.current("price"),
            np.full(18, -24),
        )

    def test__compute_deposits(self, test_firms):
        pass

    def test__compute_debt(self, test_firms):
        test_firms.ts.debt.append(np.full(18, 10.0))
        test_firms.ts.debt_installments.append(np.full(18, 0.5))
        test_firms.ts.received_credit.append(np.full(18, 3.0))
        test_firms.ts.short_term_loan_debt.append(np.full(18, 3.0))
        test_firms.ts.long_term_loan_debt.append(np.full(18, 10.0))
        assert np.allclose(test_firms.compute_debt(), np.full(18, 13.0))

    def test__compute_productivity_investment_uses_industry_good_prices(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        n_industries = test_firms.n_industries
        industry_prices = np.linspace(1.0, 2.0, n_industries)
        firm_prices = np.linspace(1.0, 2.0, n_firms * 2)
        production = np.ones(n_firms)

        test_firms.ts.price.append(firm_prices)
        test_firms.ts.production.append(production)
        test_firms.base_capital_inputs_depreciation_matrix = np.eye(n_industries)
        test_firms.update_total_newly_bought_costs(current_good_prices=industry_prices)
        test_firms.ts.total_capital_inputs_bought_costs.append(np.full(n_firms, 10.0))

        expected_replacement_cost = production * industry_prices[test_firms.states["Industry"]]
        expected_productivity_investment = np.maximum(0.0, 10.0 - expected_replacement_cost)

        assert np.allclose(test_firms.compute_productivity_investment(), expected_productivity_investment)

    def test__real_productivity_investment_uses_capital_bundle_deflator(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        n_industries = test_firms.n_industries
        industry_prices = np.linspace(1.0, 2.0, n_industries)
        nominal_investment = np.full(n_firms, 10.0)

        test_firms.current_good_prices = industry_prices
        test_firms.ts.production.append(np.ones(n_firms))
        test_firms.base_capital_inputs_depreciation_matrix = np.eye(n_industries)

        expected_deflator = industry_prices[test_firms.states["Industry"]]
        assert np.allclose(test_firms.compute_capital_bundle_deflator(), expected_deflator)
        assert np.allclose(
            test_firms.compute_real_productivity_investment(nominal_investment),
            nominal_investment / expected_deflator,
        )

    def test__real_productivity_investment_falls_when_capital_prices_rise(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        n_industries = test_firms.n_industries
        nominal_investment = np.full(n_firms, 10.0)

        test_firms.ts.production.append(np.ones(n_firms))
        test_firms.base_capital_inputs_depreciation_matrix = np.eye(n_industries)

        test_firms.current_good_prices = np.ones(n_industries)
        real_at_base_prices = test_firms.compute_real_productivity_investment(nominal_investment)

        test_firms.current_good_prices = np.full(n_industries, 2.0)
        real_at_high_prices = test_firms.compute_real_productivity_investment(nominal_investment)

        assert np.allclose(real_at_high_prices, real_at_base_prices / 2.0)
