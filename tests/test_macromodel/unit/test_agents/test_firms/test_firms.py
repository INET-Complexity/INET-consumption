import numpy as np
import pytest

from macromodel.agents.firms.firms import Firms
from macromodel.configurations.firms_configuration import FirmsConfiguration


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
            "capital_depreciation_costs",
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
            "activity_finance_available",
            "activity_finance_hard_obligations",
            "activity_finance_gap_before_revision",
            "intermediate_purchase_finance_scale",
            "capital_purchase_finance_scale",
            "technical_investment_finance_scale",
            "tfp_investment_finance_scale",
            "real_executed_productivity_investment",
            "net_capital_investment_above_replacement",
            # "real_amount_bought_as_capital_inputs",
        ]:
            assert ts_key in test_firms.ts.get_keys()

    def test__from_pickled_agent_rejects_cfc_depreciation_without_cfc_replacement(self, datawrapper):
        country = datawrapper.synthetic_countries["FRA"]
        configuration = FirmsConfiguration()
        configuration.parameters.capital_depreciation_accounting_mode = "eurostat_cfc"
        configuration.parameters.capital_replacement_matrix_source = "capital_compensation"

        with pytest.raises(ValueError, match="capital_replacement_matrix_source='eurostat_cfc_output'"):
            Firms.from_pickled_agent(
                synthetic_firms=country.firms,
                configuration=configuration,
                country_name="FRA",
                all_country_names=["FRA", "ROW"],
                goods_criticality_matrix=country.goods_criticality_matrix,
                average_initial_price=country.industry_data["industry_vectors"]["Average Initial Price"].values,
                industries=datawrapper.industries,
            )

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

    def test_invalid_capital_compensation_accounting_mode_raises(self, test_firms):
        test_firms.configuration.parameters.capital_compensation_accounting_mode = "invalid"
        test_firms.ts.used_capital_inputs_costs.append(np.full(18, 10.0))

        try:
            test_firms.compute_profits()
        except ValueError as exc:
            assert "capital_compensation_accounting_mode" in str(exc)
        else:
            raise AssertionError("Expected ValueError for invalid capital compensation accounting mode")

    def test__compute_profits_surplus_pool_excludes_capital_compensation_charge(self, test_firms):
        test_firms.configuration.parameters.capital_compensation_accounting_mode = "surplus_pool"
        test_firms.ts.used_intermediate_inputs_costs.append(np.full(18, 10.0))
        test_firms.ts.used_capital_inputs_costs.append(np.full(18, 10.0))
        test_firms.ts.taxes_paid_on_production.append(np.full(18, 1.0))
        test_firms.ts.interest_paid.append(np.full(18, 2.0))
        test_firms.ts.total_wage.append(np.full(18, 1.0))
        assert np.allclose(
            test_firms.compute_profits() - test_firms.ts.current("production") * test_firms.ts.current("price"),
            np.full(18, -14),
        )

    def test__compute_unit_costs_surplus_pool_excludes_capital_compensation_charge(self, test_firms):
        test_firms.configuration.parameters.capital_compensation_accounting_mode = "surplus_pool"
        test_firms.ts.used_intermediate_inputs_costs.append(np.full(18, 10.0))
        test_firms.ts.used_capital_inputs_costs.append(np.full(18, 10.0))
        test_firms.ts.taxes_paid_on_production.append(np.full(18, 1.0))
        test_firms.ts.total_wage.append(np.full(18, 1.0))
        expected = np.divide(
            np.full(18, 12.0),
            test_firms.ts.current("production"),
            out=np.zeros_like(test_firms.ts.current("production")),
            where=test_firms.ts.current("production") != 0.0,
        )

        assert np.allclose(test_firms.compute_unit_costs(), expected)

    def test__compute_deposits_surplus_pool_excludes_capital_compensation_charge(self, test_firms):
        test_firms.configuration.parameters.capital_compensation_accounting_mode = "surplus_pool"
        test_firms.ts.deposits.append(np.full(18, 100.0))
        test_firms.ts.nominal_amount_sold_in_lcu.append(np.full(18, 50.0))
        test_firms.ts.nominal_amount_spent_in_lcu.append(np.full((18, 18), 1.0))
        test_firms.ts.total_wage.append(np.full(18, 1.0))
        test_firms.ts.used_intermediate_inputs_costs.append(np.full(18, 10.0))
        test_firms.ts.used_capital_inputs_costs.append(np.full(18, 30.0))
        test_firms.ts.taxes_paid_on_production.append(np.full(18, 2.0))
        test_firms.ts.corporate_taxes_paid.append(np.full(18, 3.0))
        test_firms.ts.interest_paid.append(np.full(18, 4.0))
        test_firms.ts.received_credit.append(np.full(18, 5.0))
        test_firms.ts.debt_installments.append(np.full(18, 6.0))

        assert np.allclose(test_firms.compute_deposits(), np.full(18, 121.0))

    def test__compute_deposits_uses_purchase_spending_not_used_input_costs(self, test_firms):
        test_firms.configuration.parameters.capital_compensation_accounting_mode = "production_cost"
        test_firms.ts.deposits.append(np.full(18, 100.0))
        test_firms.ts.nominal_amount_sold_in_lcu.append(np.full(18, 50.0))
        test_firms.ts.nominal_amount_spent_in_lcu.append(np.full((18, 18), 1.0))
        test_firms.ts.total_wage.append(np.full(18, 1.0))
        test_firms.ts.used_intermediate_inputs_costs.append(np.full(18, 10.0))
        test_firms.ts.used_capital_inputs_costs.append(np.full(18, 30.0))
        test_firms.ts.taxes_paid_on_production.append(np.full(18, 2.0))
        test_firms.ts.corporate_taxes_paid.append(np.full(18, 3.0))
        test_firms.ts.interest_paid.append(np.full(18, 4.0))
        test_firms.ts.received_credit.append(np.full(18, 5.0))
        test_firms.ts.debt_installments.append(np.full(18, 6.0))

        assert np.allclose(test_firms.compute_deposits(), np.full(18, 121.0))

    def test__depreciation_reduces_profit_and_unit_cost_not_deposits(self, test_firms):
        test_firms.configuration.parameters.capital_compensation_accounting_mode = "surplus_pool"
        test_firms.configuration.parameters.capital_depreciation_accounting_mode = "eurostat_cfc"
        test_firms.ts.used_intermediate_inputs_costs.append(np.full(18, 10.0))
        test_firms.ts.used_capital_inputs_costs.append(np.full(18, 30.0))
        test_firms.ts.capital_depreciation_costs.append(np.full(18, 7.0))
        test_firms.ts.taxes_paid_on_production.append(np.full(18, 2.0))
        test_firms.ts.interest_paid.append(np.full(18, 4.0))
        test_firms.ts.total_wage.append(np.full(18, 1.0))

        profit_delta = test_firms.compute_profits() - test_firms.ts.current("production") * test_firms.ts.current("price")
        assert np.allclose(profit_delta, np.full(18, -24.0))

        expected_unit_costs = np.divide(
            np.full(18, 20.0),
            test_firms.ts.current("production"),
            out=np.zeros_like(test_firms.ts.current("production")),
            where=test_firms.ts.current("production") != 0.0,
        )
        assert np.allclose(test_firms.compute_unit_costs(), expected_unit_costs)

        test_firms.ts.deposits.append(np.full(18, 100.0))
        test_firms.ts.nominal_amount_sold_in_lcu.append(np.full(18, 50.0))
        test_firms.ts.nominal_amount_spent_in_lcu.append(np.full((18, 18), 1.0))
        test_firms.ts.corporate_taxes_paid.append(np.full(18, 3.0))
        test_firms.ts.received_credit.append(np.full(18, 5.0))
        test_firms.ts.debt_installments.append(np.full(18, 6.0))

        assert np.allclose(test_firms.compute_deposits(), np.full(18, 121.0))

    def test__compute_capital_depreciation_costs_uses_output_scaled_cfc(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        test_firms.configuration.parameters.capital_depreciation_accounting_mode = "eurostat_cfc"
        test_firms.capital_depreciation_rates = np.full(test_firms.n_industries, 0.1)
        test_firms.ts.override_current("production", np.full(n_firms, 10.0))
        test_firms.ts.override_current("price", np.full(n_firms, 2.0))
        test_firms.ts.override_current("capital_inputs_stock_value", np.full(n_firms, 1000.0))

        assert np.allclose(test_firms.compute_capital_depreciation_costs(), np.full(n_firms, 2.0))

    def test__equity_reflects_stocks_and_deposits_not_non_cash_depreciation(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        n_industries = test_firms.n_industries
        prices = np.linspace(1.0, 2.0, n_industries)
        intermediate_stock = np.full((n_firms, n_industries), 2.0)
        capital_stock = np.full((n_firms, n_industries), 3.0)

        test_firms.ts.override_current("inventory", np.full(n_firms, 4.0))
        test_firms.ts.override_current("price", np.full(n_firms, 5.0))
        test_firms.ts.override_current("intermediate_inputs_stock", intermediate_stock)
        test_firms.ts.override_current("capital_inputs_stock", capital_stock)
        test_firms.ts.override_current("deposits", np.full(n_firms, 100.0))
        test_firms.ts.override_current("debt", np.full(n_firms, 7.0))
        test_firms.ts.override_current("capital_depreciation_costs", np.full(n_firms, 999.0))

        expected = (
            4.0 * 5.0
            + intermediate_stock @ prices
            + capital_stock @ prices
            + 100.0
            - 7.0
        )
        assert np.allclose(test_firms.compute_equity(prices), expected)

        depleted_capital_stock = capital_stock.copy()
        depleted_capital_stock[:, 0] -= 1.0
        test_firms.ts.override_current("capital_inputs_stock", depleted_capital_stock)

        assert np.allclose(test_firms.compute_equity(prices), expected - prices[0])

    def test__compute_debt(self, test_firms):
        test_firms.ts.debt.append(np.full(18, 10.0))
        test_firms.ts.debt_installments.append(np.full(18, 0.5))
        test_firms.ts.received_credit.append(np.full(18, 3.0))
        test_firms.ts.short_term_loan_debt.append(np.full(18, 3.0))
        test_firms.ts.long_term_loan_debt.append(np.full(18, 10.0))
        assert np.allclose(test_firms.compute_debt(), np.full(18, 13.0))

    def test__compute_net_capital_investment_uses_industry_good_prices(self, test_firms):
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
        expected_net_capital_investment = np.maximum(0.0, 10.0 - expected_replacement_cost)

        assert np.allclose(
            test_firms.compute_net_capital_investment_above_replacement(),
            expected_net_capital_investment,
        )

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

    def test__execute_productivity_investment_splits_net_capital_from_planner_execution(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        n_industries = test_firms.n_industries

        test_firms.current_good_prices = np.ones(n_industries)
        test_firms.ts.production.append(np.ones(n_firms))
        test_firms.base_capital_inputs_depreciation_matrix = np.zeros((n_industries, n_industries))
        test_firms.ts.total_capital_inputs_bought_costs.append(np.full(n_firms, 10.0))
        test_firms.ts.planned_productivity_investment.append(np.zeros(n_firms))
        test_firms.ts.planned_tfp_investment.append(np.zeros(n_firms))
        test_firms.ts.planned_technical_investment.append(np.zeros((n_firms, n_industries)))

        test_firms.execute_productivity_investment()

        assert test_firms.ts.current("net_capital_investment_above_replacement").sum() > 0
        assert np.allclose(test_firms.ts.current("executed_productivity_investment"), 0.0)
        assert np.allclose(test_firms.ts.current("executed_tfp_investment"), 0.0)
        assert np.allclose(test_firms.ts.current("executed_technical_investment"), 0.0)
        assert np.allclose(test_firms.ts.current("real_executed_productivity_investment"), 0.0)

    def test__post_credit_activity_revision_none_mode_leaves_targets_unchanged(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        n_industries = test_firms.n_industries
        target_intermediate = np.full((n_firms, n_industries), 2.0)
        target_capital = np.full((n_firms, n_industries), 3.0)
        planned_technical = np.full((n_firms, n_industries), 4.0)
        planned_tfp = np.full(n_firms, 5.0)

        test_firms.configuration.parameters.firm_activity_finance_revision_mode = "none"
        test_firms.ts.override_current("target_intermediate_inputs", target_intermediate.copy())
        test_firms.ts.override_current("target_capital_inputs", target_capital.copy())
        test_firms.ts.override_current("planned_technical_investment", planned_technical.copy())
        test_firms.ts.override_current("planned_tfp_investment", planned_tfp.copy())

        test_firms.revise_activity_against_available_finance(
            expected_lcu_prices=np.ones(n_industries),
            wage_obligation_preview=np.zeros(n_firms),
            production_tax_obligation_preview=np.zeros(n_firms),
        )

        assert np.allclose(test_firms.ts.current("target_intermediate_inputs"), target_intermediate)
        assert np.allclose(test_firms.ts.current("target_capital_inputs"), target_capital)
        assert np.allclose(test_firms.ts.current("planned_technical_investment"), planned_technical)
        assert np.allclose(test_firms.ts.current("planned_tfp_investment"), planned_tfp)
        assert np.allclose(test_firms.ts.current("intermediate_purchase_finance_scale"), 1.0)
        assert np.allclose(test_firms.ts.current("activity_finance_gap_before_revision"), 0.0)

    def test__post_credit_activity_revision_zero_finance_zeroes_discretionary_plans(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        n_industries = test_firms.n_industries
        test_firms.configuration.parameters.firm_activity_finance_revision_mode = "post_credit_cash_budget"
        test_firms.ts.override_current("deposits", np.full(n_firms, -10.0))
        test_firms.ts.override_current("received_short_term_credit", np.zeros(n_firms))
        test_firms.ts.override_current("received_long_term_credit", np.zeros(n_firms))
        test_firms.ts.override_current("interest_paid", np.zeros(n_firms))
        test_firms.ts.override_current("debt_installments", np.zeros(n_firms))
        test_firms.ts.override_current("target_intermediate_inputs", np.full((n_firms, n_industries), 2.0))
        test_firms.ts.override_current("target_capital_inputs", np.full((n_firms, n_industries), 3.0))
        test_firms.ts.override_current("planned_technical_investment", np.full((n_firms, n_industries), 4.0))
        test_firms.ts.override_current("planned_tfp_investment", np.full(n_firms, 5.0))

        test_firms.revise_activity_against_available_finance(
            expected_lcu_prices=np.ones(n_industries),
            wage_obligation_preview=np.zeros(n_firms),
            production_tax_obligation_preview=np.zeros(n_firms),
        )

        assert np.allclose(test_firms.ts.current("activity_finance_available"), 0.0)
        assert np.allclose(test_firms.ts.current("target_intermediate_inputs"), 0.0)
        assert np.allclose(test_firms.ts.current("target_capital_inputs"), 0.0)
        assert np.allclose(test_firms.ts.current("planned_technical_investment"), 0.0)
        assert np.allclose(test_firms.ts.current("planned_tfp_investment"), 0.0)

    def test__post_credit_activity_revision_prioritises_inputs_and_scales_vectors(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        n_industries = test_firms.n_industries
        target_intermediate = np.zeros((n_firms, n_industries))
        target_capital = np.zeros((n_firms, n_industries))
        planned_technical = np.zeros((n_firms, n_industries))
        planned_tfp = np.zeros(n_firms)
        target_intermediate[0, :2] = [4.0, 2.0]
        target_capital[0, :2] = [10.0, 20.0]
        planned_technical[0, :2] = [6.0, 3.0]
        planned_tfp[0] = 7.0

        test_firms.configuration.parameters.firm_activity_finance_revision_mode = "post_credit_cash_budget"
        test_firms.ts.override_current("deposits", np.zeros(n_firms))
        test_firms.ts.override_current("received_short_term_credit", np.r_[21.0, np.zeros(n_firms - 1)])
        test_firms.ts.override_current("received_long_term_credit", np.zeros(n_firms))
        test_firms.ts.override_current("interest_paid", np.zeros(n_firms))
        test_firms.ts.override_current("debt_installments", np.zeros(n_firms))
        test_firms.ts.override_current("target_intermediate_inputs", target_intermediate.copy())
        test_firms.ts.override_current("target_capital_inputs", target_capital.copy())
        test_firms.ts.override_current("planned_technical_investment", planned_technical.copy())
        test_firms.ts.override_current("planned_tfp_investment", planned_tfp.copy())
        unconstrained = np.full((n_firms, n_industries), 99.0)
        test_firms.ts.override_current("unconstrained_target_intermediate_inputs", unconstrained.copy())

        test_firms.revise_activity_against_available_finance(
            expected_lcu_prices=np.ones(n_industries),
            wage_obligation_preview=np.zeros(n_firms),
            production_tax_obligation_preview=np.zeros(n_firms),
        )

        assert np.allclose(test_firms.ts.current("target_intermediate_inputs")[0, :2], [4.0, 2.0])
        assert np.allclose(test_firms.ts.current("capital_purchase_finance_scale")[0], 0.5)
        assert np.allclose(test_firms.ts.current("target_capital_inputs")[0, :2], [5.0, 10.0])
        assert np.allclose(test_firms.ts.current("planned_technical_investment")[0], 0.0)
        assert np.allclose(test_firms.ts.current("planned_tfp_investment")[0], 0.0)
        assert np.allclose(test_firms.ts.current("unconstrained_target_intermediate_inputs"), unconstrained)

    def test__post_credit_activity_revision_uses_technical_investment_as_nominal_budget(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        n_industries = test_firms.n_industries
        target_intermediate = np.zeros((n_firms, n_industries))
        target_capital = np.zeros((n_firms, n_industries))
        planned_technical = np.zeros((n_firms, n_industries))
        planned_tfp = np.zeros(n_firms)
        planned_technical[0, :2] = [6.0, 3.0]
        planned_tfp[0] = 4.0

        test_firms.configuration.parameters.firm_activity_finance_revision_mode = "post_credit_cash_budget"
        test_firms.ts.override_current("deposits", np.zeros(n_firms))
        test_firms.ts.override_current("received_short_term_credit", np.r_[12.0, np.zeros(n_firms - 1)])
        test_firms.ts.override_current("received_long_term_credit", np.zeros(n_firms))
        test_firms.ts.override_current("interest_paid", np.zeros(n_firms))
        test_firms.ts.override_current("debt_installments", np.zeros(n_firms))
        test_firms.ts.override_current("target_intermediate_inputs", target_intermediate)
        test_firms.ts.override_current("target_capital_inputs", target_capital)
        test_firms.ts.override_current("planned_technical_investment", planned_technical.copy())
        test_firms.ts.override_current("planned_tfp_investment", planned_tfp.copy())

        test_firms.revise_activity_against_available_finance(
            expected_lcu_prices=np.full(n_industries, 3.0),
            wage_obligation_preview=np.zeros(n_firms),
            production_tax_obligation_preview=np.zeros(n_firms),
        )

        assert np.isclose(test_firms.ts.current("planned_technical_investment_expected_costs")[0], 9.0)
        assert np.allclose(test_firms.ts.current("planned_technical_investment")[0, :2], [6.0, 3.0])
        assert np.isclose(test_firms.ts.current("planned_tfp_investment")[0], 3.0)
        assert np.isclose(test_firms.ts.current("planned_productivity_investment")[0], 12.0)

    def test__post_credit_activity_revision_hard_obligations_reduce_available_finance(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        n_industries = test_firms.n_industries
        test_firms.configuration.parameters.firm_activity_finance_revision_mode = "post_credit_cash_budget"
        test_firms.ts.override_current("deposits", np.r_[20.0, np.zeros(n_firms - 1)])
        test_firms.ts.override_current("received_short_term_credit", np.zeros(n_firms))
        test_firms.ts.override_current("received_long_term_credit", np.zeros(n_firms))
        test_firms.ts.override_current("interest_paid", np.r_[2.0, np.zeros(n_firms - 1)])
        test_firms.ts.override_current("debt_installments", np.r_[3.0, np.zeros(n_firms - 1)])
        test_firms.ts.override_current("target_intermediate_inputs", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("target_capital_inputs", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("planned_technical_investment", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("planned_tfp_investment", np.zeros(n_firms))

        test_firms.revise_activity_against_available_finance(
            expected_lcu_prices=np.ones(n_industries),
            wage_obligation_preview=np.r_[4.0, np.zeros(n_firms - 1)],
            production_tax_obligation_preview=np.r_[1.0, np.zeros(n_firms - 1)],
        )

        assert np.isclose(test_firms.ts.current("activity_finance_hard_obligations")[0], 10.0)
        assert np.isclose(test_firms.ts.current("activity_finance_available")[0], 10.0)

    def test__wage_and_tax_previews_do_not_append_time_series(self, test_firms):
        wage_len = len(test_firms.ts.total_wage)
        tax_len = len(test_firms.ts.taxes_paid_on_production)

        test_firms.compute_total_wage_obligation(
            corresponding_firm=np.arange(test_firms.ts.current("n_firms")),
            individual_wages=np.ones(test_firms.ts.current("n_firms")),
            income_taxes=0.2,
            employee_social_insurance_tax=0.05,
            employer_social_insurance_tax=0.03,
            cpi=1.0,
        )
        test_firms.compute_taxes_paid_on_production(taxes_less_subsidies_rates=np.zeros(test_firms.n_industries))

        assert len(test_firms.ts.total_wage) == wage_len
        assert len(test_firms.ts.taxes_paid_on_production) == tax_len

    def test__prepare_buying_goods_uses_expected_prices_and_revised_targets(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        n_industries = test_firms.n_industries
        previous_prices = np.full(n_industries, 2.0)
        expected_inflation = 0.5
        test_firms.configuration.parameters.firm_activity_finance_revision_mode = "post_credit_cash_budget"
        test_firms.ts.override_current("deposits", np.zeros(n_firms))
        test_firms.ts.override_current("received_short_term_credit", np.zeros(n_firms))
        test_firms.ts.override_current("received_long_term_credit", np.zeros(n_firms))
        test_firms.ts.override_current("interest_paid", np.zeros(n_firms))
        test_firms.ts.override_current("debt_installments", np.zeros(n_firms))
        test_firms.ts.override_current("planned_technical_investment", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("planned_tfp_investment", np.zeros(n_firms))

        test_firms.prepare_buying_goods(
            previous_good_prices=previous_prices,
            expected_inflation=expected_inflation,
            assume_zero_growth=True,
            wage_obligation_preview=np.zeros(n_firms),
            production_tax_obligation_preview=np.zeros(n_firms),
        )

        planned_cost = (
            (
                test_firms.ts.initial("target_intermediate_inputs")
                + test_firms.ts.initial("target_capital_inputs")
            )
            * (previous_prices * (1 + expected_inflation))[None, :]
        ).sum(axis=1)
        assert np.allclose(test_firms.ts.current("planned_intermediate_purchase_expected_costs")
                           + test_firms.ts.current("planned_capital_purchase_expected_costs"), planned_cost)
        assert np.allclose(test_firms.transactor_buyer_states["Initial Goods"], 0.0)

    def test__prepare_buying_goods_converts_nominal_technical_investment_to_real_goods(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        n_industries = test_firms.n_industries
        previous_prices = np.full(n_industries, 2.0)
        expected_inflation = 0.5
        planned_technical = np.zeros((n_firms, n_industries))
        planned_technical[0, :2] = [6.0, 3.0]

        test_firms.configuration.parameters.firm_activity_finance_revision_mode = "none"
        test_firms.ts.override_current("planned_technical_investment", planned_technical)

        test_firms.prepare_buying_goods(
            previous_good_prices=previous_prices,
            expected_inflation=expected_inflation,
            assume_zero_growth=True,
            wage_obligation_preview=np.zeros(n_firms),
            production_tax_obligation_preview=np.zeros(n_firms),
        )

        expected_goods = test_firms.ts.initial("target_intermediate_inputs") + test_firms.ts.initial(
            "target_capital_inputs"
        )
        expected_goods[0, :2] += [2.0, 1.0]
        assert np.allclose(test_firms.transactor_buyer_states["Initial Goods"], expected_goods)

    def test__compute_tfp_growth_uses_nominal_tfp_investment_over_nominal_output(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        production = np.full(n_firms, 10.0)
        prices = np.full(n_firms, 2.0)
        executed_tfp_investment = np.full(n_firms, 1.0)

        class TFPGrowthSpy:
            def __init__(self):
                self.kwargs = None

            def compute_tfp_growth(self, **kwargs):
                self.kwargs = kwargs
                return np.zeros_like(kwargs["current_tfp"])

        spy = TFPGrowthSpy()
        test_firms.functions["productivity_growth"] = spy
        test_firms.ts.production.append(production)
        test_firms.ts.price.append(prices)
        test_firms.ts.executed_tfp_investment.append(executed_tfp_investment)

        test_firms.compute_tfp_growth()

        assert np.allclose(spy.kwargs["productivity_investment"], executed_tfp_investment)
        assert np.allclose(spy.kwargs["output_value"], prices * production)
