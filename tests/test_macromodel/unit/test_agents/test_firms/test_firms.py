import numpy as np
import pytest

from macromodel.agents.firms.firms import Firms
from macromodel.agents.firms.func.production import CriticalAndImportantLeontief
from macromodel.configurations.firms_configuration import FirmsConfiguration
from macromodel.markets.credit_market.credit_market import CreditMarket
from macromodel.markets.credit_market.func.clearing import compute_firm_borrower_credit_room


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
            "pricing_mc",
            "pricing_mc_smooth",
            "pricing_ac",
            "pricing_ac_smooth",
            "pricing_material_mc",
            "pricing_labour_mc",
            "pricing_depreciation_unit_cost",
            "pricing_initial_price_gap",
            "pricing_normal_output",
            "pricing_markup_mu",
            "pricing_markup_base_mu",
            "pricing_markup_lower",
            "pricing_markup_upper",
            "pricing_markup_residual_factor",
            "pricing_markup_residual_status",
            "pricing_ac_floor_binding",
            "pricing_ac_fallback_binding",
            "pricing_gate_state",
            "pricing_fallback_code",
            "pricing_cost_normalization_factor",
            "pricing_cost_normalization_raw_gap",
            "pricing_cost_normalization_status",
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
            "credit_budget_internal_cash",
            "credit_budget_existing_overdraft",
            "expected_sales",
            "credit_budget_wage_obligations",
            "credit_budget_production_tax_obligations",
            "credit_budget_corporate_tax_obligations",
            "credit_budget_interest_obligations",
            "credit_budget_debt_installments",
            "credit_budget_hard_obligations",
            "credit_budget_cash_after_hard_obligations",
            "credit_budget_available_after_hard_and_overdraft",
            "credit_budget_intermediate_costs",
            "credit_budget_tfp_costs",
            "credit_budget_working_capital_budget",
            "credit_budget_capital_costs",
            "credit_budget_technical_investment_costs",
            "credit_budget_investment_budget",
            "credit_budget_remaining_internal_finance_after_working_capital",
            "target_short_term_credit",
            "target_debt_rollover_credit",
            "target_overdraft_refinance_credit",
            "ordinary_target_short_term_credit",
            "target_long_term_credit",
            "received_short_term_credit",
            "received_debt_rollover_credit",
            "received_overdraft_refinance_credit",
            "received_ordinary_short_term_credit",
            "received_long_term_credit",
            "received_credit",
            "firm_settlement_available_cash_before_debt_service",
            "firm_settlement_corporate_tax_reserve",
            "firm_settlement_cash_after_tax_reserve",
            "firm_settlement_opening_interest_arrears",
            "firm_settlement_scheduled_interest_due",
            "firm_settlement_contractual_interest_due",
            "firm_settlement_payable_interest",
            "firm_settlement_closing_interest_arrears",
            "firm_settlement_unpaid_interest",
            "firm_settlement_capitalized_interest",
            "firm_settlement_opening_principal_arrears",
            "firm_settlement_scheduled_principal_due",
            "firm_settlement_contractual_principal_due",
            "firm_settlement_payable_principal",
            "firm_settlement_closing_principal_arrears",
            "firm_settlement_unpaid_principal",
            "firm_settlement_debt_rollover_shortfall",
            "firm_settlement_overdraft_refinance_used",
            "firm_settlement_overdraft_refinance_shortfall",
            "firm_settlement_residual_overdraft_exposure",
            "firm_settlement_illiquid_flag",
            "firm_settlement_default_flag",
            "firm_settlement_balance_sheet_residual",
            "firm_settlement_transaction_flow_residual",
            "firm_settlement_accounting_control_passed",
            "total_credit_exposure",
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
            "activity_finance_opening_deposits",
            "activity_finance_feasible_target_production",
            "activity_finance_feasible_desired_labour_inputs",
            "activity_finance_feasibility_residual",
            "activity_finance_realised_feasible_target_production",
            "activity_finance_realised_labour_scale",
            "intermediate_purchase_finance_scale",
            "capital_purchase_finance_scale",
            "technical_investment_finance_scale",
            "tfp_investment_finance_scale",
            "real_executed_productivity_investment",
            "net_capital_investment_above_replacement",
            "direct_tfp_investment_cash_expense",
            # "real_amount_bought_as_capital_inputs",
        ]:
            assert ts_key in test_firms.ts.get_keys()

    def test__pricing_cost_state_does_not_bootstrap_from_accounting_unit_costs(self, test_firms):
        assert np.isfinite(test_firms.ts.current("unit_costs")).any()
        assert np.isnan(test_firms.ts.current("pricing_mc_smooth")).all()
        assert np.isnan(test_firms.ts.current("pricing_ac_smooth")).all()

    def test__compute_price_passes_initial_output_weights_and_appends_normalization_diagnostics(self, test_firms):
        class CapturingPriceSetter:
            def compute_price(self, **kwargs):
                self.initial_output_weights = kwargs["initial_output_weights"].copy()
                shape = kwargs["prev_prices"].shape
                self.last_pricing_cost_normalization_factor = np.full(shape, 1.25)
                self.last_pricing_cost_normalization_raw_gap = np.full(shape, 1.10)
                self.last_pricing_cost_normalization_status = np.full(shape, 1.0)
                return kwargs["prev_prices"].copy()

        price_setter = CapturingPriceSetter()
        test_firms.functions["prices"] = price_setter
        previous_average_good_prices = np.ones(test_firms.n_industries)

        test_firms.compute_price(
            current_estimated_ppi_inflation=0.0,
            previous_average_good_prices=previous_average_good_prices,
            ppi_during=np.ones(2),
        )

        np.testing.assert_allclose(price_setter.initial_output_weights, test_firms.ts.initial("production"))
        np.testing.assert_allclose(test_firms.ts.current("pricing_cost_normalization_factor"), 1.25)
        np.testing.assert_allclose(test_firms.ts.current("pricing_cost_normalization_raw_gap"), 1.10)
        np.testing.assert_allclose(test_firms.ts.current("pricing_cost_normalization_status"), 1.0)

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

    def test__from_pickled_agent_rejects_stale_cfc_output_ratio_cache(self, datawrapper):
        country = datawrapper.synthetic_countries["FRA"]
        country.firms.capital_depreciation_rate_basis = "output"
        configuration = FirmsConfiguration()
        configuration.parameters.capital_depreciation_accounting_mode = "eurostat_cfc"
        configuration.parameters.capital_replacement_matrix_source = "eurostat_cfc_output"

        with pytest.raises(ValueError, match="capital_depreciation_rate_basis='capital_stock'"):
            Firms.from_pickled_agent(
                synthetic_firms=country.firms,
                configuration=configuration,
                country_name="FRA",
                all_country_names=["FRA", "ROW"],
                goods_criticality_matrix=country.goods_criticality_matrix,
                average_initial_price=country.industry_data["industry_vectors"]["Average Initial Price"].values,
                industries=datawrapper.industries,
            )

    def test__borrower_credit_room_is_not_clipped_by_target_credit(self, test_banks, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        test_banks.parameters.enable_firm_loans_return_on_assets_restriction = False
        test_banks.parameters.enable_firm_loans_return_on_equity_restriction = False
        test_banks.parameters.enable_firm_loans_dscr_restriction = False
        test_banks.parameters.firm_loans_capital_stock_collateral_ratio = 0.5
        test_firms.ts.override_current("capital_inputs_stock_value", np.full(n_firms, 100.0))
        test_firms.ts.override_current("debt", np.full(n_firms, 10.0))
        test_firms.ts.override_current("deposits", np.zeros(n_firms))
        test_firms.ts.override_current("target_long_term_credit", np.zeros(n_firms))
        test_firms.ts.override_current("target_short_term_credit", np.zeros(n_firms))

        room = compute_firm_borrower_credit_room(
            banks=test_banks,
            firms=test_firms,
            allow_short_term_firm_loans=False,
        )

        assert np.allclose(room["short_term"], 0.0)
        assert np.allclose(room["long_term"], 40.0)
        assert np.allclose(room["total"], 40.0)

        room_with_short_term_first = compute_firm_borrower_credit_room(
            banks=test_banks,
            firms=test_firms,
            allow_short_term_firm_loans=True,
        )

        assert np.allclose(room_with_short_term_first["short_term"], 40.0)
        assert np.allclose(room_with_short_term_first["long_term"], 0.0)
        assert np.allclose(room_with_short_term_first["total"], 40.0)

    def test__append_excess_demand_finance_potential_diagnostics_records_cap_share(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        before = len(test_firms.ts.excess_demand_finance_cash)
        test_firms.ts.override_current("activity_finance_opening_deposits", np.full(n_firms, 10.0))
        test_firms.ts.override_current("target_debt_rollover_credit", np.zeros(n_firms))
        test_firms.ts.override_current("target_overdraft_refinance_credit", np.zeros(n_firms))

        q_sold = np.zeros(n_firms)
        q_excess = np.zeros(n_firms)
        q_excess[0] = 5.0
        test_firms.append_excess_demand_finance_potential_diagnostics(
            expected_lcu_prices=np.ones(test_firms.n_industries),
            wage_obligation_preview=np.zeros(n_firms),
            non_loan_interest_obligation_preview=np.full(n_firms, 2.0),
            borrower_st_credit_room=np.full(n_firms, 3.0),
            borrower_lt_credit_room=np.full(n_firms, 7.0),
            borrower_total_credit_room=np.full(n_firms, 10.0),
            q_sold=q_sold,
            q_excess=q_excess,
        )

        assert len(test_firms.ts.excess_demand_finance_cash) == before + 1
        assert np.allclose(test_firms.ts.current("excess_demand_finance_cash"), 8.0)
        assert np.allclose(test_firms.ts.current("excess_demand_borrower_max_credit"), 10.0)
        assert np.allclose(test_firms.ts.current("excess_demand_activity_finance_borrower"), 18.0)
        assert np.all(test_firms.ts.current("excess_demand_potential_capacity_borrower") >= 0.0)
        assert np.isfinite(test_firms.ts.current("excess_demand_above_borrower_cap_share")[0])
        assert np.isnan(test_firms.ts.current("excess_demand_above_borrower_cap_share")[1:]).all()
        assert np.isnan(test_firms.ts.current("excess_demand_potential_capacity_supply")).all()
        assert np.isnan(test_firms.ts.current("excess_demand_above_supply_cap_share")).all()
        assert test_firms._last_excess_demand_finance_potential is None

    def test__append_excess_demand_finance_potential_diagnostics_does_not_reuse_stale_cache(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        test_firms.ts.override_current("activity_finance_opening_deposits", np.full(n_firms, 10.0))
        test_firms.ts.override_current("target_debt_rollover_credit", np.zeros(n_firms))
        test_firms.ts.override_current("target_overdraft_refinance_credit", np.zeros(n_firms))

        kwargs = {
            "expected_lcu_prices": np.ones(test_firms.n_industries),
            "wage_obligation_preview": np.zeros(n_firms),
            "non_loan_interest_obligation_preview": np.zeros(n_firms),
            "borrower_st_credit_room": np.zeros(n_firms),
            "borrower_lt_credit_room": np.zeros(n_firms),
            "q_sold": np.zeros(n_firms),
            "q_excess": np.zeros(n_firms),
        }

        test_firms.append_excess_demand_finance_potential_diagnostics(
            **kwargs,
            borrower_total_credit_room=np.full(n_firms, 1.0),
        )
        test_firms.append_excess_demand_finance_potential_diagnostics(
            **kwargs,
            borrower_total_credit_room=np.full(n_firms, 7.0),
        )

        assert np.allclose(test_firms.ts.current("excess_demand_borrower_max_credit"), 7.0)
        assert test_firms._last_excess_demand_finance_potential is None

    def test__compute_excess_demand_finance_potential_capacities_rescales_by_bank_supply(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        test_firms.ts.override_current("activity_finance_opening_deposits", np.zeros(n_firms))
        test_firms.ts.override_current("target_debt_rollover_credit", np.zeros(n_firms))
        test_firms.ts.override_current("target_overdraft_refinance_credit", np.zeros(n_firms))

        test_firms.compute_excess_demand_finance_potential_capacities(
            expected_lcu_prices=np.ones(test_firms.n_industries),
            wage_obligation_preview=np.zeros(n_firms),
            non_loan_interest_obligation_preview=np.zeros(n_firms),
            borrower_st_credit_room=np.full(n_firms, 3.0),
            borrower_lt_credit_room=np.full(n_firms, 7.0),
            borrower_total_credit_room=np.full(n_firms, 10.0),
            q_sold=np.zeros(n_firms),
            ordinary_bank_supply=5.0 * n_firms,
        )

        cache = test_firms._last_excess_demand_finance_potential
        assert np.allclose(cache["borrower_max_credit"], 10.0)
        assert np.allclose(cache["supply_max_credit"], 5.0)
        assert np.all(test_firms.get_supply_adjusted_excess_demand_capacity() >= 0.0)

        q_excess = np.zeros(n_firms)
        q_excess[0] = test_firms.get_supply_adjusted_excess_demand_capacity()[0] + 1.0
        test_firms.append_cached_excess_demand_finance_potential_diagnostics(q_excess=q_excess)

        assert np.allclose(test_firms.ts.current("excess_demand_supply_max_credit"), 5.0)
        assert np.allclose(test_firms.ts.current("excess_demand_activity_finance_supply"), 5.0)
        assert np.all(test_firms.ts.current("excess_demand_potential_capacity_supply") >= 0.0)
        assert np.isfinite(test_firms.ts.current("excess_demand_above_supply_cap_share")[0])

    def test__compute_excess_demand_finance_potential_capacities_rejects_bad_bank_supply(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")

        with pytest.raises(ValueError, match="ordinary_bank_supply"):
            test_firms.compute_excess_demand_finance_potential_capacities(
                expected_lcu_prices=np.ones(test_firms.n_industries),
                wage_obligation_preview=np.zeros(n_firms),
                non_loan_interest_obligation_preview=np.zeros(n_firms),
                borrower_st_credit_room=np.zeros(n_firms),
                borrower_lt_credit_room=np.zeros(n_firms),
                borrower_total_credit_room=np.zeros(n_firms),
                q_sold=np.zeros(n_firms),
                ordinary_bank_supply=np.nan,
            )

    @staticmethod
    def _set_simple_activity_solver_inputs(
        test_firms,
        target_production: float = 20.0,
        include_capital: bool = False,
    ) -> None:
        n_firms = test_firms.ts.current("n_firms")
        n_industries = test_firms.n_industries
        firm_industry = test_firms.states["Industry"][0]
        intermediate_coefficients = np.zeros((n_industries, n_industries))
        intermediate_coefficients[0, firm_industry] = 1.0
        capital_use = np.zeros((n_industries, n_industries))
        if include_capital:
            capital_use[0, firm_industry] = 1.0
        test_firms.base_intermediate_inputs_productivity_matrix = intermediate_coefficients
        test_firms.base_capital_input_use_matrix = capital_use
        test_firms.ts.override_current("target_production", np.r_[target_production, np.zeros(n_firms - 1)])
        test_firms.ts.override_current("tfp_multiplier", np.ones(n_firms))
        test_firms.ts.override_current("limiting_intermediate_inputs", np.full(n_firms, 1_000.0))
        test_firms.ts.override_current("limiting_capital_inputs", np.full(n_firms, 1_000.0))
        test_firms.ts.override_current("intermediate_inputs_stock", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("capital_inputs_stock", np.zeros((n_firms, n_industries)))

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

    def test__critical_and_important_leontief_uses_current_bought_inputs_without_nan_propagation(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        n_industries = test_firms.n_industries
        firm_index = 0
        output_industry = test_firms.states["Industry"][firm_index]
        critical_input = output_industry
        noncritical_input = (output_industry + 1) % n_industries

        production = np.zeros(n_firms)
        production[firm_index] = 10.0
        coefficients = np.zeros((n_industries, n_industries))
        coefficients[critical_input, output_industry] = 2.0
        coefficients[noncritical_input, output_industry] = 5.0
        criticality = np.ones((n_firms, n_industries))
        criticality[firm_index, noncritical_input] = 0.0
        opening_stock = np.zeros((n_firms, n_industries))
        opening_stock[firm_index, critical_input] = 5.0
        opening_stock[firm_index, noncritical_input] = 0.25
        bought_inputs = np.full((n_firms, n_industries), np.nan)
        bought_inputs[firm_index, critical_input] = 0.0
        bought_inputs[firm_index, noncritical_input] = 1.25
        current_good_prices = np.ones(n_industries)
        current_good_prices[critical_input] = 3.0
        current_good_prices[noncritical_input] = 7.0

        test_firms.functions["production"] = CriticalAndImportantLeontief()
        test_firms.base_intermediate_inputs_productivity_matrix = coefficients
        test_firms.goods_criticality_matrix = criticality
        test_firms.ts.override_current("production", production)
        test_firms.ts.override_current("intermediate_inputs_stock", opening_stock)
        test_firms.ts.override_current("real_amount_bought_as_intermediate_inputs", bought_inputs)

        used_inputs = test_firms.compute_used_intermediate_inputs()
        test_firms.ts.override_current("used_intermediate_inputs", used_inputs)
        used_costs = test_firms.compute_used_intermediate_inputs_costs(current_good_prices)
        closing_stock = test_firms.compute_intermediate_inputs_stock()

        expected_used = np.zeros((n_firms, n_industries))
        expected_used[firm_index, critical_input] = 5.0
        expected_used[firm_index, noncritical_input] = 1.5

        assert not np.isnan(used_inputs).any()
        assert not np.isnan(closing_stock).any()
        assert np.allclose(used_inputs, expected_used)
        assert np.allclose(used_costs[firm_index], 25.5)
        assert np.allclose(used_costs[np.arange(n_firms) != firm_index], 0.0)
        assert np.allclose(closing_stock, np.zeros((n_firms, n_industries)))

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

    def test__compute_profits_subtracts_direct_tfp_cash_expense(self, test_firms):
        test_firms.configuration.parameters.capital_compensation_accounting_mode = "surplus_pool"
        test_firms.ts.used_intermediate_inputs_costs.append(np.full(18, 10.0))
        test_firms.ts.used_capital_inputs_costs.append(np.full(18, 10.0))
        test_firms.ts.taxes_paid_on_production.append(np.full(18, 1.0))
        test_firms.ts.interest_paid.append(np.full(18, 2.0))
        test_firms.ts.total_wage.append(np.full(18, 1.0))
        test_firms.ts.direct_tfp_investment_cash_expense.append(np.full(18, 4.0))

        assert np.allclose(
            test_firms.compute_profits() - test_firms.ts.current("production") * test_firms.ts.current("price"),
            np.full(18, -18),
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

    def test__compute_deposits_subtracts_direct_tfp_cash_expense_once(self, test_firms):
        test_firms.ts.deposits.append(np.full(18, 100.0))
        test_firms.ts.nominal_amount_sold_in_lcu.append(np.full(18, 50.0))
        test_firms.ts.nominal_amount_spent_in_lcu.append(np.full((18, 18), 1.0))
        test_firms.ts.total_wage.append(np.full(18, 1.0))
        test_firms.ts.taxes_paid_on_production.append(np.full(18, 2.0))
        test_firms.ts.corporate_taxes_paid.append(np.full(18, 3.0))
        test_firms.ts.interest_paid.append(np.full(18, 4.0))
        test_firms.ts.received_credit.append(np.full(18, 5.0))
        test_firms.ts.debt_installments.append(np.full(18, 6.0))
        test_firms.ts.direct_tfp_investment_cash_expense.append(np.full(18, 7.0))

        assert np.allclose(test_firms.compute_deposits(), np.full(18, 114.0))

    def test__depreciation_reduces_profit_and_unit_cost_not_deposits(self, test_firms):
        test_firms.configuration.parameters.capital_compensation_accounting_mode = "surplus_pool"
        test_firms.configuration.parameters.capital_depreciation_accounting_mode = "eurostat_cfc"
        test_firms.ts.used_intermediate_inputs_costs.append(np.full(18, 10.0))
        test_firms.ts.used_capital_inputs_costs.append(np.full(18, 30.0))
        test_firms.ts.capital_depreciation_costs.append(np.full(18, 7.0))
        test_firms.ts.taxes_paid_on_production.append(np.full(18, 2.0))
        test_firms.ts.interest_paid.append(np.full(18, 4.0))
        test_firms.ts.total_wage.append(np.full(18, 1.0))

        profit_delta = test_firms.compute_profits() - test_firms.ts.current("production") * test_firms.ts.current(
            "price"
        )
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

    def test__direct_tfp_cash_expense_uses_executed_firm_paid_tfp(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        n_industries = test_firms.n_industries

        class CapitalBoundPlanner:
            executes_direct_tfp_independently = False

        test_firms.functions["productivity_investment_planner"] = CapitalBoundPlanner()
        test_firms.current_good_prices = np.ones(n_industries)
        test_firms.base_capital_input_use_matrix = np.zeros((n_industries, n_industries))
        test_firms.ts.production.append(np.ones(n_firms))
        test_firms.ts.total_capital_inputs_bought_costs.append(np.full(n_firms, 5.0))
        test_firms.ts.planned_productivity_investment.append(np.full(n_firms, 10.0))
        test_firms.ts.planned_tfp_investment.append(np.full(n_firms, 8.0))
        test_firms.ts.planned_technical_investment.append(np.zeros((n_firms, n_industries)))

        test_firms.execute_productivity_investment()

        assert np.allclose(test_firms.ts.current("executed_tfp_investment"), 4.0)
        assert np.allclose(test_firms.ts.current("direct_tfp_investment_cash_expense"), 4.0)

    def test__compute_capital_depreciation_costs_uses_stock_value_cfc(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        test_firms.configuration.parameters.capital_depreciation_accounting_mode = "eurostat_cfc"
        test_firms.capital_depreciation_rates = np.full(test_firms.n_industries, 0.1)
        test_firms.ts.override_current("production", np.full(n_firms, 10.0))
        test_firms.ts.override_current("price", np.full(n_firms, 2.0))
        test_firms.ts.override_current("capital_inputs_stock_value", np.full(n_firms, 1000.0))

        assert np.allclose(test_firms.compute_capital_depreciation_costs(), np.full(n_firms, 100.0))

    def test__pricing_material_mc_uses_reciprocal_productivity(self, test_firms):
        n_industries = test_firms.n_industries
        test_firms.base_intermediate_inputs_productivity_matrix = np.zeros((n_industries, n_industries))
        test_firms.base_intermediate_inputs_productivity_matrix[0, :] = 2.0
        test_firms.base_intermediate_inputs_productivity_matrix[1, :] = 4.0
        good_prices = np.full(n_industries, np.nan)
        good_prices[0] = 8.0

        material_mc = test_firms.compute_pricing_material_mc(good_prices)

        assert np.allclose(material_mc, 8.0 / 2.0 + 8.0 / 4.0)

    def test__pricing_material_mc_is_invalid_when_all_good_prices_are_invalid(self, test_firms):
        material_mc = test_firms.compute_pricing_material_mc(np.full(test_firms.n_industries, np.nan))

        assert np.isnan(material_mc).all()

    def test__pricing_normal_output_uses_conservative_capital_floor(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        test_firms.pricing_rho_k_by_sector = np.full(test_firms.n_industries, 2.0)
        capital_stock = np.zeros((n_firms, test_firms.n_industries))
        capital_stock[:, 0] = 100.0
        target_production = np.full(n_firms, 1.0)
        target_production[0] = 60.0
        test_firms.ts.override_current("capital_inputs_stock", capital_stock)
        test_firms.ts.override_current("target_production", target_production)

        normal_output = test_firms.compute_pricing_normal_output_candidate(capital_floor_lambda=0.25)

        assert normal_output[0] == pytest.approx(60.0)
        assert np.allclose(normal_output[1:], 50.0)

    def test__pricing_depreciation_unit_cost_allocates_cfc_over_normal_output(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        test_firms.configuration.parameters.capital_depreciation_accounting_mode = "eurostat_cfc"
        test_firms.ts.override_current("capital_depreciation_costs", np.full(n_firms, 30.0))
        pricing_normal_output = np.full(n_firms, 10.0)
        pricing_normal_output[0] = 15.0

        depreciation_unit_cost = test_firms.compute_pricing_depreciation_unit_cost(pricing_normal_output)

        assert depreciation_unit_cost[0] == pytest.approx(2.0)
        assert np.allclose(depreciation_unit_cost[1:], 3.0)

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

        expected = 4.0 * 5.0 + intermediate_stock @ prices + capital_stock @ prices + 100.0 - 7.0
        assert np.allclose(test_firms.compute_equity(prices), expected)

        depleted_capital_stock = capital_stock.copy()
        depleted_capital_stock[:, 0] -= 1.0
        test_firms.ts.override_current("capital_inputs_stock", depleted_capital_stock)

        assert np.allclose(test_firms.compute_equity(prices), expected - prices[0])

    def test__check_firm_accounting_controls_passes_on_consistent_settlement_state(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        n_industries = test_firms.n_industries

        opening_deposits = np.full(n_firms, 100.0)
        nominal_amount_sold_in_lcu = np.full(n_firms, 20.0)
        received_credit = np.full(n_firms, 10.0)
        total_wage = np.full(n_firms, 5.0)
        nominal_amount_spent_in_lcu = np.zeros((n_firms, n_industries))
        direct_tfp_investment_cash_expense = np.zeros(n_firms)
        taxes_paid_on_production = np.full(n_firms, 2.0)
        corporate_taxes_paid = np.full(n_firms, 3.0)
        interest_paid = np.full(n_firms, 4.0)
        debt_installments = np.full(n_firms, 6.0)
        closing_deposits = (
            opening_deposits
            + nominal_amount_sold_in_lcu
            + received_credit
            - total_wage
            - nominal_amount_spent_in_lcu.sum(axis=1)
            - direct_tfp_investment_cash_expense
            - taxes_paid_on_production
            - corporate_taxes_paid
            - interest_paid
            - debt_installments
        )
        current_good_prices = np.ones(n_industries)

        test_firms.ts.override_current("activity_finance_opening_deposits", opening_deposits)
        test_firms.ts.override_current("deposits", closing_deposits)
        test_firms.ts.override_current("nominal_amount_sold_in_lcu", nominal_amount_sold_in_lcu)
        test_firms.ts.override_current("received_credit", received_credit)
        test_firms.ts.override_current("total_wage", total_wage)
        test_firms.ts.override_current("nominal_amount_spent_in_lcu", nominal_amount_spent_in_lcu)
        test_firms.ts.override_current("direct_tfp_investment_cash_expense", direct_tfp_investment_cash_expense)
        test_firms.ts.override_current("taxes_paid_on_production", taxes_paid_on_production)
        test_firms.ts.override_current("corporate_taxes_paid", corporate_taxes_paid)
        test_firms.ts.override_current("interest_paid", interest_paid)
        test_firms.ts.override_current("debt_installments", debt_installments)
        test_firms.ts.override_current("inventory", np.ones(n_firms))
        test_firms.ts.override_current("price", np.full(n_firms, 2.0))
        test_firms.ts.override_current("intermediate_inputs_stock", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("capital_inputs_stock", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("debt", np.full(n_firms, 5.0))
        test_firms.ts.override_current("equity", np.full(n_firms, 107.0))

        result = test_firms.check_firm_accounting_controls(
            current_good_prices=current_good_prices,
            enforce=True,
        )

        assert np.allclose(result["balance_sheet_residual"], 0.0)
        assert np.allclose(result["transaction_flow_residual"], 0.0)
        assert np.all(result["control_passed"])
        assert np.allclose(test_firms.ts.current("firm_settlement_balance_sheet_residual"), 0.0)
        assert np.allclose(test_firms.ts.current("firm_settlement_transaction_flow_residual"), 0.0)
        assert np.all(test_firms.ts.current("firm_settlement_accounting_control_passed"))

    def test__check_firm_accounting_controls_allows_high_scale_roundoff(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        n_industries = test_firms.n_industries
        opening_deposits = np.full(n_firms, 1e24)
        closing_deposits = opening_deposits + 1e8

        test_firms.ts.override_current("activity_finance_opening_deposits", opening_deposits)
        test_firms.ts.override_current("deposits", closing_deposits)
        test_firms.ts.override_current("nominal_amount_sold_in_lcu", np.zeros(n_firms))
        test_firms.ts.override_current("received_credit", np.zeros(n_firms))
        test_firms.ts.override_current("total_wage", np.zeros(n_firms))
        test_firms.ts.override_current("nominal_amount_spent_in_lcu", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("direct_tfp_investment_cash_expense", np.zeros(n_firms))
        test_firms.ts.override_current("taxes_paid_on_production", np.zeros(n_firms))
        test_firms.ts.override_current("corporate_taxes_paid", np.zeros(n_firms))
        test_firms.ts.override_current("interest_paid", np.zeros(n_firms))
        test_firms.ts.override_current("debt_installments", np.zeros(n_firms))
        test_firms.ts.override_current("inventory", np.zeros(n_firms))
        test_firms.ts.override_current("price", np.ones(n_firms))
        test_firms.ts.override_current("intermediate_inputs_stock", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("capital_inputs_stock", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("debt", np.zeros(n_firms))
        test_firms.ts.override_current("equity", closing_deposits)

        result = test_firms.check_firm_accounting_controls(
            current_good_prices=np.ones(n_industries),
            enforce=True,
        )

        assert np.max(np.abs(result["transaction_flow_residual"])) > 1e-4
        assert np.all(result["control_passed"])

    def test__check_firm_accounting_controls_raises_on_mismatch(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        n_industries = test_firms.n_industries
        test_firms.ts.override_current("activity_finance_opening_deposits", np.full(n_firms, 100.0))
        test_firms.ts.override_current("deposits", np.full(n_firms, 110.0))
        test_firms.ts.override_current("nominal_amount_sold_in_lcu", np.full(n_firms, 20.0))
        test_firms.ts.override_current("received_credit", np.full(n_firms, 10.0))
        test_firms.ts.override_current("total_wage", np.full(n_firms, 5.0))
        test_firms.ts.override_current("nominal_amount_spent_in_lcu", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("direct_tfp_investment_cash_expense", np.zeros(n_firms))
        test_firms.ts.override_current("taxes_paid_on_production", np.full(n_firms, 2.0))
        test_firms.ts.override_current("corporate_taxes_paid", np.full(n_firms, 3.0))
        test_firms.ts.override_current("interest_paid", np.full(n_firms, 4.0))
        test_firms.ts.override_current("debt_installments", np.full(n_firms, 6.0))
        test_firms.ts.override_current("inventory", np.ones(n_firms))
        test_firms.ts.override_current("price", np.full(n_firms, 2.0))
        test_firms.ts.override_current("intermediate_inputs_stock", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("capital_inputs_stock", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("debt", np.full(n_firms, 5.0))
        test_firms.ts.override_current("equity", np.full(n_firms, 108.0))

        with pytest.raises(ValueError, match="Firm accounting control violation"):
            test_firms.check_firm_accounting_controls(
                current_good_prices=np.ones(n_industries),
                enforce=True,
            )

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
        test_firms.base_capital_input_use_matrix = np.eye(n_industries)
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
        test_firms.base_capital_input_use_matrix = np.eye(n_industries)

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
        test_firms.base_capital_input_use_matrix = np.eye(n_industries)

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
        test_firms.base_capital_input_use_matrix = np.zeros((n_industries, n_industries))
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

    def test__post_credit_activity_revision_constrained_solver_satisfies_budget(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        n_industries = test_firms.n_industries
        target_intermediate = np.zeros((n_firms, n_industries))
        planned_technical = np.zeros((n_firms, n_industries))
        planned_tfp = np.zeros(n_firms)
        target_intermediate[0, 0] = 20.0
        planned_technical[0, 0] = 5.0
        planned_tfp[0] = 5.0

        self._set_simple_activity_solver_inputs(test_firms, target_production=20.0)
        test_firms.configuration.parameters.firm_activity_finance_revision_mode = "post_credit_cash_budget"
        test_firms.ts.override_current("deposits", np.zeros(n_firms))
        test_firms.ts.override_current("received_short_term_credit", np.r_[15.0, np.zeros(n_firms - 1)])
        test_firms.ts.override_current("received_long_term_credit", np.zeros(n_firms))
        test_firms.ts.override_current("interest_paid", np.zeros(n_firms))
        test_firms.ts.override_current("debt_installments", np.zeros(n_firms))
        test_firms.ts.override_current("target_intermediate_inputs", target_intermediate.copy())
        test_firms.ts.override_current("target_capital_inputs", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("planned_technical_investment", planned_technical.copy())
        test_firms.ts.override_current("planned_tfp_investment", planned_tfp.copy())
        unconstrained = np.full((n_firms, n_industries), 99.0)
        test_firms.ts.override_current("unconstrained_target_intermediate_inputs", unconstrained.copy())

        test_firms.revise_activity_against_available_finance(
            expected_lcu_prices=np.ones(n_industries),
            wage_obligation_preview=np.zeros(n_firms),
            production_tax_obligation_preview=np.zeros(n_firms),
        )

        expected_intermediate = 15.0 / 1.5
        assert np.isclose(test_firms.ts.current("target_intermediate_inputs")[0, 0], expected_intermediate)
        assert np.isclose(test_firms.ts.current("planned_technical_investment")[0, 0], expected_intermediate * 0.25)
        assert np.isclose(test_firms.ts.current("planned_tfp_investment")[0], expected_intermediate * 0.25)
        assert test_firms.ts.current("activity_finance_feasibility_residual")[0] >= -1e-6
        assert np.isclose(
            test_firms.ts.current("activity_finance_feasible_target_production")[0], expected_intermediate
        )
        assert np.isclose(test_firms.ts.current("intermediate_purchase_finance_scale")[0], 0.5, atol=1e-6)
        assert np.allclose(test_firms.ts.current("unconstrained_target_intermediate_inputs"), unconstrained)

    def test__post_credit_activity_revision_constrained_solver_includes_wage_costs(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        n_industries = test_firms.n_industries
        target_intermediate = np.zeros((n_firms, n_industries))
        target_intermediate[0, 0] = 20.0

        self._set_simple_activity_solver_inputs(test_firms, target_production=20.0)
        test_firms.configuration.parameters.firm_activity_finance_revision_mode = "post_credit_cash_budget"
        test_firms.ts.override_current("deposits", np.r_[15.0, np.zeros(n_firms - 1)])
        test_firms.ts.override_current("received_short_term_credit", np.zeros(n_firms))
        test_firms.ts.override_current("received_long_term_credit", np.zeros(n_firms))
        test_firms.ts.override_current("interest_paid", np.zeros(n_firms))
        test_firms.ts.override_current("debt_installments", np.zeros(n_firms))
        test_firms.ts.override_current("labour_inputs", np.r_[20.0, np.ones(n_firms - 1)])
        test_firms.ts.override_current("target_intermediate_inputs", target_intermediate.copy())
        test_firms.ts.override_current("target_capital_inputs", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("planned_technical_investment", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("planned_tfp_investment", np.zeros(n_firms))

        test_firms.revise_activity_against_available_finance(
            expected_lcu_prices=np.ones(n_industries),
            wage_obligation_preview=np.r_[10.0, np.zeros(n_firms - 1)],
            production_tax_obligation_preview=np.zeros(n_firms),
        )

        assert np.isclose(test_firms.ts.current("activity_finance_feasible_target_production")[0], 10.0, atol=1e-5)
        assert np.isclose(test_firms.ts.current("target_intermediate_inputs")[0, 0], 10.0, atol=1e-5)
        assert test_firms.ts.current("activity_finance_feasibility_residual")[0] >= -1e-6
        assert np.isclose(test_firms.ts.current("activity_finance_gap_before_revision")[0], 15.0)

    def test__post_credit_activity_revision_unconstrained_keeps_original_plan(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        n_industries = test_firms.n_industries
        target_intermediate = np.zeros((n_firms, n_industries))
        target_capital = np.zeros((n_firms, n_industries))
        planned_technical = np.zeros((n_firms, n_industries))
        planned_tfp = np.zeros(n_firms)
        target_intermediate[0, 0] = 0.0
        target_capital[0, 1] = 3.0
        planned_technical[0, :2] = [6.0, 3.0]
        planned_tfp[0] = 7.0

        self._set_simple_activity_solver_inputs(test_firms, target_production=20.0)
        test_firms.configuration.parameters.firm_activity_finance_revision_mode = "post_credit_cash_budget"
        test_firms.ts.override_current("deposits", np.r_[1_000.0, np.zeros(n_firms - 1)])
        test_firms.ts.override_current("received_short_term_credit", np.zeros(n_firms))
        test_firms.ts.override_current("received_long_term_credit", np.zeros(n_firms))
        test_firms.ts.override_current("interest_paid", np.zeros(n_firms))
        test_firms.ts.override_current("debt_installments", np.zeros(n_firms))
        test_firms.ts.override_current("target_intermediate_inputs", target_intermediate.copy())
        test_firms.ts.override_current("target_capital_inputs", target_capital.copy())
        test_firms.ts.override_current("planned_technical_investment", planned_technical.copy())
        test_firms.ts.override_current("planned_tfp_investment", planned_tfp.copy())
        test_firms.ts.override_current("planned_productivity_investment", planned_tfp + planned_technical.sum(axis=1))

        test_firms.revise_activity_against_available_finance(
            expected_lcu_prices=np.ones(n_industries),
            wage_obligation_preview=np.zeros(n_firms),
            production_tax_obligation_preview=np.zeros(n_firms),
        )

        assert np.allclose(test_firms.ts.current("target_intermediate_inputs"), target_intermediate)
        assert np.allclose(test_firms.ts.current("target_capital_inputs"), target_capital)
        assert np.allclose(test_firms.ts.current("planned_technical_investment"), planned_technical)
        assert np.allclose(test_firms.ts.current("planned_tfp_investment"), planned_tfp)
        assert np.isclose(test_firms.ts.current("activity_finance_feasible_target_production")[0], 20.0)
        assert np.allclose(test_firms.ts.current("intermediate_purchase_finance_scale"), 1.0)

    def test__post_credit_activity_revision_feasible_output_is_bounded_by_target_production(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        n_industries = test_firms.n_industries
        target_production = np.zeros(n_firms)
        target_production[:3] = [5.0, 10.0, 0.0]

        test_firms.configuration.parameters.firm_activity_finance_revision_mode = "post_credit_cash_budget"
        test_firms.ts.override_current("target_production", target_production.copy())
        test_firms.ts.override_current("tfp_multiplier", np.ones(n_firms))
        test_firms.ts.override_current("limiting_intermediate_inputs", np.full(n_firms, 1_000.0))
        test_firms.ts.override_current("limiting_capital_inputs", np.full(n_firms, 1_000.0))
        test_firms.ts.override_current("deposits", np.full(n_firms, 1_000.0))
        test_firms.ts.override_current("received_short_term_credit", np.zeros(n_firms))
        test_firms.ts.override_current("received_long_term_credit", np.zeros(n_firms))
        test_firms.ts.override_current("interest_paid", np.zeros(n_firms))
        test_firms.ts.override_current("debt_installments", np.zeros(n_firms))
        test_firms.ts.override_current("target_intermediate_inputs", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("target_capital_inputs", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("planned_technical_investment", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("planned_tfp_investment", np.zeros(n_firms))

        test_firms.revise_activity_against_available_finance(
            expected_lcu_prices=np.ones(n_industries),
            wage_obligation_preview=np.zeros(n_firms),
            production_tax_obligation_preview=np.zeros(n_firms),
        )

        feasible_y = test_firms.ts.current("activity_finance_feasible_target_production")
        assert np.all(feasible_y >= 0.0)
        assert np.all(feasible_y <= target_production + 1e-12)
        assert np.allclose(feasible_y, target_production)

    def test__post_credit_activity_revision_feasible_output_is_monotone_in_finance(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        n_industries = test_firms.n_industries
        target_intermediate = np.zeros((n_firms, n_industries))
        target_intermediate[0, 0] = 20.0

        self._set_simple_activity_solver_inputs(test_firms, target_production=20.0)
        test_firms.configuration.parameters.firm_activity_finance_revision_mode = "post_credit_cash_budget"
        test_firms.ts.override_current("received_long_term_credit", np.zeros(n_firms))
        test_firms.ts.override_current("interest_paid", np.zeros(n_firms))
        test_firms.ts.override_current("debt_installments", np.zeros(n_firms))
        test_firms.ts.override_current("target_capital_inputs", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("planned_technical_investment", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("planned_tfp_investment", np.zeros(n_firms))

        feasible_outputs = []
        for finance in [5.0, 10.0]:
            test_firms.ts.override_current("deposits", np.r_[finance, np.zeros(n_firms - 1)])
            test_firms.ts.override_current("received_short_term_credit", np.zeros(n_firms))
            test_firms.ts.override_current("target_intermediate_inputs", target_intermediate.copy())
            test_firms.revise_activity_against_available_finance(
                expected_lcu_prices=np.ones(n_industries),
                wage_obligation_preview=np.zeros(n_firms),
                production_tax_obligation_preview=np.zeros(n_firms),
            )
            feasible_outputs.append(test_firms.ts.current("activity_finance_feasible_target_production")[0])

        assert feasible_outputs[1] >= feasible_outputs[0]

    def test__post_credit_activity_revision_uses_target_production_as_activity_cap(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        n_industries = test_firms.n_industries
        target_intermediate = np.zeros((n_firms, n_industries))
        target_intermediate[0, 0] = 100.0
        unconstrained = np.full((n_firms, n_industries), 200.0)

        self._set_simple_activity_solver_inputs(test_firms, target_production=5.0)
        test_firms.configuration.parameters.firm_activity_finance_revision_mode = "post_credit_cash_budget"
        test_firms.ts.override_current("deposits", np.r_[30.0, np.zeros(n_firms - 1)])
        test_firms.ts.override_current("received_short_term_credit", np.zeros(n_firms))
        test_firms.ts.override_current("received_long_term_credit", np.zeros(n_firms))
        test_firms.ts.override_current("interest_paid", np.zeros(n_firms))
        test_firms.ts.override_current("debt_installments", np.zeros(n_firms))
        test_firms.ts.override_current("target_intermediate_inputs", target_intermediate.copy())
        test_firms.ts.override_current("target_capital_inputs", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("planned_technical_investment", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("planned_tfp_investment", np.zeros(n_firms))
        test_firms.ts.override_current("unconstrained_target_intermediate_inputs", unconstrained.copy())

        test_firms.revise_activity_against_available_finance(
            expected_lcu_prices=np.ones(n_industries),
            wage_obligation_preview=np.zeros(n_firms),
            production_tax_obligation_preview=np.zeros(n_firms),
        )

        feasible_y = test_firms.ts.current("activity_finance_feasible_target_production")
        assert np.isclose(feasible_y[0], 5.0, atol=1e-5)
        assert np.all(feasible_y <= test_firms.ts.current("target_production") + 1e-12)
        assert np.isclose(test_firms.ts.current("target_intermediate_inputs")[0, 0], 5.0, atol=1e-5)
        assert np.allclose(test_firms.ts.current("unconstrained_target_intermediate_inputs"), unconstrained)

    def test__post_credit_activity_revision_uses_candidate_y_for_intermediate_stock_buffer(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        n_industries = test_firms.n_industries
        target_intermediate = np.zeros((n_firms, n_industries))
        target_intermediate[0, 0] = 10.0
        initial_stock = np.zeros((n_firms, n_industries))
        current_stock = np.zeros((n_firms, n_industries))
        initial_stock[0, 0] = 2.0
        current_stock[0, 0] = 3.0

        self._set_simple_activity_solver_inputs(test_firms, target_production=5.0)
        test_firms.functions["target_intermediate_inputs"].target_intermediate_inputs_fraction = 1.0
        test_firms.ts.dicts["production"][0] = np.r_[10.0, np.ones(n_firms - 1)]
        test_firms.ts.dicts["intermediate_inputs_stock"][0] = initial_stock
        test_firms.ts.intermediate_inputs_stock.append(current_stock)
        test_firms.configuration.parameters.firm_activity_finance_revision_mode = "post_credit_cash_budget"
        test_firms.ts.override_current("deposits", np.r_[3.0, np.zeros(n_firms - 1)])
        test_firms.ts.override_current("received_short_term_credit", np.zeros(n_firms))
        test_firms.ts.override_current("received_long_term_credit", np.zeros(n_firms))
        test_firms.ts.override_current("interest_paid", np.zeros(n_firms))
        test_firms.ts.override_current("debt_installments", np.zeros(n_firms))
        test_firms.ts.override_current("target_intermediate_inputs", target_intermediate.copy())
        test_firms.ts.override_current("target_capital_inputs", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("planned_technical_investment", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("planned_tfp_investment", np.zeros(n_firms))

        test_firms.revise_activity_against_available_finance(
            expected_lcu_prices=np.ones(n_industries),
            wage_obligation_preview=np.zeros(n_firms),
            production_tax_obligation_preview=np.zeros(n_firms),
        )

        assert np.isclose(test_firms.ts.current("activity_finance_feasible_target_production")[0], 5.0)
        assert np.isclose(test_firms.ts.current("target_intermediate_inputs")[0, 0], 3.0)

    def test__post_credit_activity_revision_uses_candidate_y_for_capital_stock_buffer(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        n_industries = test_firms.n_industries
        target_capital = np.zeros((n_firms, n_industries))
        target_capital[0, 0] = 10.0
        initial_stock = np.zeros((n_firms, n_industries))
        current_stock = np.zeros((n_firms, n_industries))
        initial_stock[0, 0] = 2.0
        current_stock[0, 0] = 3.0

        self._set_simple_activity_solver_inputs(test_firms, target_production=5.0, include_capital=True)
        test_firms.base_intermediate_inputs_productivity_matrix = np.zeros((n_industries, n_industries))
        test_firms.functions["target_capital_inputs"].target_capital_inputs_fraction = 1.0
        test_firms.ts.dicts["production"][0] = np.r_[10.0, np.ones(n_firms - 1)]
        test_firms.ts.dicts["capital_inputs_stock"][0] = initial_stock
        test_firms.ts.capital_inputs_stock.append(current_stock)
        test_firms.configuration.parameters.firm_activity_finance_revision_mode = "post_credit_cash_budget"
        test_firms.ts.override_current("deposits", np.r_[3.0, np.zeros(n_firms - 1)])
        test_firms.ts.override_current("received_short_term_credit", np.zeros(n_firms))
        test_firms.ts.override_current("received_long_term_credit", np.zeros(n_firms))
        test_firms.ts.override_current("interest_paid", np.zeros(n_firms))
        test_firms.ts.override_current("debt_installments", np.zeros(n_firms))
        test_firms.ts.override_current("target_intermediate_inputs", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("target_capital_inputs", target_capital.copy())
        test_firms.ts.override_current("planned_technical_investment", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("planned_tfp_investment", np.zeros(n_firms))

        test_firms.revise_activity_against_available_finance(
            expected_lcu_prices=np.ones(n_industries),
            wage_obligation_preview=np.zeros(n_firms),
            production_tax_obligation_preview=np.zeros(n_firms),
        )

        assert np.isclose(test_firms.ts.current("activity_finance_feasible_target_production")[0], 5.0)
        assert np.isclose(test_firms.ts.current("target_capital_inputs")[0, 0], 3.0)

    def test__post_credit_activity_revision_does_not_revise_labour_or_production(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        n_industries = test_firms.n_industries
        target_production = np.linspace(1.0, float(n_firms), n_firms)
        production = target_production / 2.0
        labour_inputs = target_production / 3.0

        test_firms.configuration.parameters.firm_activity_finance_revision_mode = "post_credit_cash_budget"
        test_firms.ts.override_current("deposits", np.zeros(n_firms))
        test_firms.ts.override_current("received_short_term_credit", np.zeros(n_firms))
        test_firms.ts.override_current("received_long_term_credit", np.zeros(n_firms))
        test_firms.ts.override_current("interest_paid", np.zeros(n_firms))
        test_firms.ts.override_current("debt_installments", np.zeros(n_firms))
        test_firms.ts.override_current("target_production", target_production.copy())
        test_firms.ts.override_current("production", production.copy())
        test_firms.ts.override_current("labour_inputs", labour_inputs.copy())
        test_firms.ts.override_current("target_intermediate_inputs", np.ones((n_firms, n_industries)))
        test_firms.ts.override_current("target_capital_inputs", np.ones((n_firms, n_industries)))
        test_firms.ts.override_current("planned_technical_investment", np.ones((n_firms, n_industries)))
        test_firms.ts.override_current("planned_tfp_investment", np.ones(n_firms))

        test_firms.revise_activity_against_available_finance(
            expected_lcu_prices=np.ones(n_industries),
            wage_obligation_preview=np.zeros(n_firms),
            production_tax_obligation_preview=np.zeros(n_firms),
        )

        assert np.allclose(test_firms.ts.current("target_production"), target_production)
        assert np.allclose(test_firms.ts.current("production"), production)
        assert np.allclose(test_firms.ts.current("labour_inputs"), labour_inputs)

    def test__post_credit_activity_revision_repairs_overdraft_before_activity(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        n_industries = test_firms.n_industries
        target_intermediate = np.zeros((n_firms, n_industries))
        target_intermediate[0, 0] = 20.0

        self._set_simple_activity_solver_inputs(test_firms, target_production=20.0)
        test_firms.configuration.parameters.firm_activity_finance_revision_mode = "post_credit_cash_budget"
        test_firms.ts.override_current("deposits", np.r_[-10.0, np.zeros(n_firms - 1)])
        test_firms.ts.override_current("received_short_term_credit", np.r_[25.0, np.zeros(n_firms - 1)])
        test_firms.ts.override_current("received_long_term_credit", np.zeros(n_firms))
        test_firms.ts.override_current("interest_paid", np.zeros(n_firms))
        test_firms.ts.override_current("debt_installments", np.zeros(n_firms))
        test_firms.ts.override_current("target_intermediate_inputs", target_intermediate.copy())
        test_firms.ts.override_current("target_capital_inputs", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("planned_technical_investment", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("planned_tfp_investment", np.zeros(n_firms))

        test_firms.revise_activity_against_available_finance(
            expected_lcu_prices=np.ones(n_industries),
            wage_obligation_preview=np.zeros(n_firms),
            production_tax_obligation_preview=np.zeros(n_firms),
        )

        assert np.isclose(test_firms.ts.current("activity_finance_available")[0], 15.0)
        assert np.isclose(test_firms.ts.current("intermediate_purchase_finance_scale")[0], 0.75)
        assert np.isclose(test_firms.ts.current("target_intermediate_inputs")[0, 0], 15.0)

    def test__post_credit_activity_revision_partial_refinance_does_not_unlock_activity(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        n_industries = test_firms.n_industries
        target_intermediate = np.zeros((n_firms, n_industries))
        target_intermediate[0, 0] = 20.0

        self._set_simple_activity_solver_inputs(test_firms, target_production=20.0)
        test_firms.configuration.parameters.firm_activity_finance_revision_mode = "post_credit_cash_budget"
        test_firms.ts.override_current("deposits", np.r_[-10.0, np.zeros(n_firms - 1)])
        test_firms.ts.override_current("received_short_term_credit", np.r_[8.0, np.zeros(n_firms - 1)])
        test_firms.ts.override_current("received_long_term_credit", np.zeros(n_firms))
        test_firms.ts.override_current("interest_paid", np.zeros(n_firms))
        test_firms.ts.override_current("debt_installments", np.zeros(n_firms))
        test_firms.ts.override_current("target_intermediate_inputs", target_intermediate.copy())
        test_firms.ts.override_current("target_capital_inputs", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("planned_technical_investment", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("planned_tfp_investment", np.zeros(n_firms))

        test_firms.revise_activity_against_available_finance(
            expected_lcu_prices=np.ones(n_industries),
            wage_obligation_preview=np.zeros(n_firms),
            production_tax_obligation_preview=np.zeros(n_firms),
        )

        assert np.isclose(test_firms.ts.current("activity_finance_available")[0], 0.0)
        assert np.isclose(test_firms.ts.current("intermediate_purchase_finance_scale")[0], 0.0)
        assert np.isclose(test_firms.ts.current("target_intermediate_inputs")[0, 0], 0.0)

    def test__post_credit_activity_revision_excludes_rollover_and_refinance_buckets(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        n_industries = test_firms.n_industries
        target_intermediate = np.zeros((n_firms, n_industries))
        target_intermediate[0, 0] = 20.0

        self._set_simple_activity_solver_inputs(test_firms, target_production=20.0)
        test_firms.configuration.parameters.firm_activity_finance_revision_mode = "post_credit_cash_budget"
        test_firms.ts.override_current("deposits", np.zeros(n_firms))
        test_firms.ts.override_current("received_short_term_credit", np.r_[15.0, np.zeros(n_firms - 1)])
        test_firms.ts.override_current("received_debt_rollover_credit", np.r_[8.0, np.zeros(n_firms - 1)])
        test_firms.ts.override_current("received_overdraft_refinance_credit", np.r_[7.0, np.zeros(n_firms - 1)])
        test_firms.ts.override_current("received_ordinary_short_term_credit", np.zeros(n_firms))
        test_firms.ts.override_current("received_long_term_credit", np.zeros(n_firms))
        test_firms.ts.override_current("interest_paid", np.zeros(n_firms))
        test_firms.ts.override_current("debt_installments", np.zeros(n_firms))
        test_firms.ts.override_current("target_intermediate_inputs", target_intermediate.copy())
        test_firms.ts.override_current("target_capital_inputs", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("planned_technical_investment", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("planned_tfp_investment", np.zeros(n_firms))

        test_firms.revise_activity_against_available_finance(
            expected_lcu_prices=np.ones(n_industries),
            wage_obligation_preview=np.zeros(n_firms),
            production_tax_obligation_preview=np.zeros(n_firms),
        )

        assert np.isclose(test_firms.ts.current("activity_finance_available")[0], 0.0)
        assert np.isclose(test_firms.ts.current("target_intermediate_inputs")[0, 0], 0.0)

    def test__post_credit_activity_revision_does_not_double_count_funded_debt_service(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        n_industries = test_firms.n_industries
        target_intermediate = np.zeros((n_firms, n_industries))
        target_intermediate[0, 0] = 20.0

        self._set_simple_activity_solver_inputs(test_firms, target_production=20.0)
        test_firms.configuration.parameters.firm_activity_finance_revision_mode = "post_credit_cash_budget"
        test_firms.ts.override_current("deposits", np.r_[20.0, np.zeros(n_firms - 1)])
        test_firms.ts.override_current("received_short_term_credit", np.r_[15.0, np.zeros(n_firms - 1)])
        test_firms.ts.override_current("received_debt_rollover_credit", np.r_[15.0, np.zeros(n_firms - 1)])
        test_firms.ts.override_current("received_overdraft_refinance_credit", np.zeros(n_firms))
        test_firms.ts.override_current("received_ordinary_short_term_credit", np.zeros(n_firms))
        test_firms.ts.override_current("received_long_term_credit", np.zeros(n_firms))
        test_firms.ts.override_current("target_intermediate_inputs", target_intermediate.copy())
        test_firms.ts.override_current("target_capital_inputs", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("planned_technical_investment", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("planned_tfp_investment", np.zeros(n_firms))

        test_firms.revise_activity_against_available_finance(
            expected_lcu_prices=np.ones(n_industries),
            wage_obligation_preview=np.zeros(n_firms),
            production_tax_obligation_preview=np.zeros(n_firms),
            interest_obligation_preview=np.r_[5.0, np.zeros(n_firms - 1)],
            loan_interest_obligation_preview=np.r_[5.0, np.zeros(n_firms - 1)],
            debt_installment_preview=np.r_[10.0, np.zeros(n_firms - 1)],
        )

        assert np.isclose(test_firms.ts.current("activity_finance_hard_obligations")[0], 0.0)
        assert np.isclose(test_firms.ts.current("activity_finance_available")[0], 20.0)
        assert np.isclose(test_firms.ts.current("target_intermediate_inputs")[0, 0], 20.0)

    def test__post_credit_activity_revision_unfunded_debt_service_squeezes_activity(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        n_industries = test_firms.n_industries
        target_intermediate = np.zeros((n_firms, n_industries))
        target_intermediate[0, 0] = 20.0

        self._set_simple_activity_solver_inputs(test_firms, target_production=20.0)
        test_firms.configuration.parameters.firm_activity_finance_revision_mode = "post_credit_cash_budget"
        test_firms.ts.override_current("deposits", np.r_[20.0, np.zeros(n_firms - 1)])
        test_firms.ts.override_current("received_short_term_credit", np.r_[5.0, np.zeros(n_firms - 1)])
        test_firms.ts.override_current("received_debt_rollover_credit", np.r_[5.0, np.zeros(n_firms - 1)])
        test_firms.ts.override_current("received_overdraft_refinance_credit", np.zeros(n_firms))
        test_firms.ts.override_current("received_ordinary_short_term_credit", np.zeros(n_firms))
        test_firms.ts.override_current("received_long_term_credit", np.zeros(n_firms))
        test_firms.ts.override_current("target_intermediate_inputs", target_intermediate.copy())
        test_firms.ts.override_current("target_capital_inputs", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("planned_technical_investment", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("planned_tfp_investment", np.zeros(n_firms))

        test_firms.revise_activity_against_available_finance(
            expected_lcu_prices=np.ones(n_industries),
            wage_obligation_preview=np.zeros(n_firms),
            production_tax_obligation_preview=np.zeros(n_firms),
            interest_obligation_preview=np.r_[5.0, np.zeros(n_firms - 1)],
            loan_interest_obligation_preview=np.r_[5.0, np.zeros(n_firms - 1)],
            debt_installment_preview=np.r_[10.0, np.zeros(n_firms - 1)],
        )

        assert np.isclose(test_firms.ts.current("activity_finance_hard_obligations")[0], 10.0)
        assert np.isclose(test_firms.ts.current("activity_finance_available")[0], 10.0)
        assert np.isclose(test_firms.ts.current("target_intermediate_inputs")[0, 0], 10.0)

    def test__post_credit_activity_revision_zero_intermediate_plan_rules_out_constrained_investment(
        self,
        test_firms,
    ):
        n_firms = test_firms.ts.current("n_firms")
        n_industries = test_firms.n_industries
        target_intermediate = np.zeros((n_firms, n_industries))
        target_capital = np.zeros((n_firms, n_industries))
        planned_technical = np.zeros((n_firms, n_industries))
        planned_tfp = np.zeros(n_firms)
        planned_technical[0, :2] = [6.0, 3.0]
        planned_tfp[0] = 4.0

        self._set_simple_activity_solver_inputs(test_firms, target_production=20.0)
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
        assert np.allclose(test_firms.ts.current("planned_technical_investment")[0], 0.0)
        assert np.isclose(test_firms.ts.current("planned_tfp_investment")[0], 0.0)
        assert np.isclose(test_firms.ts.current("planned_productivity_investment")[0], 0.0)

    def test__post_credit_activity_revision_excludes_wage_and_production_tax_from_hard_obligations(
        self,
        test_firms,
    ):
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

        assert np.isclose(test_firms.ts.current("activity_finance_hard_obligations")[0], 5.0)
        assert np.isclose(test_firms.ts.current("activity_finance_available")[0], 15.0)

    def test__post_credit_activity_revision_excludes_corporate_tax_preview_from_hard_obligations(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        n_industries = test_firms.n_industries
        test_firms.configuration.parameters.firm_activity_finance_revision_mode = "post_credit_cash_budget"
        test_firms.ts.override_current("deposits", np.r_[20.0, np.zeros(n_firms - 1)])
        test_firms.ts.override_current("received_short_term_credit", np.zeros(n_firms))
        test_firms.ts.override_current("received_long_term_credit", np.zeros(n_firms))
        test_firms.ts.override_current("interest_paid", np.zeros(n_firms))
        test_firms.ts.override_current("debt_installments", np.zeros(n_firms))
        test_firms.ts.override_current("target_intermediate_inputs", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("target_capital_inputs", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("planned_technical_investment", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("planned_tfp_investment", np.zeros(n_firms))

        test_firms.revise_activity_against_available_finance(
            expected_lcu_prices=np.ones(n_industries),
            wage_obligation_preview=np.r_[4.0, np.zeros(n_firms - 1)],
            production_tax_obligation_preview=np.r_[1.0, np.zeros(n_firms - 1)],
            corporate_tax_obligation_preview=np.r_[6.0, np.zeros(n_firms - 1)],
        )

        assert np.isclose(test_firms.ts.current("activity_finance_hard_obligations")[0], 0.0)
        assert np.isclose(test_firms.ts.current("activity_finance_available")[0], 20.0)

    def test__inventory_net_target_stays_bounded_through_feasible_activity_sequence(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        n_industries = test_firms.n_industries
        firm_industry = test_firms.states["Industry"][0]
        intermediate_coefficients = np.zeros((n_industries, n_industries))
        intermediate_coefficients[0, firm_industry] = 1.0

        test_firms.base_intermediate_inputs_productivity_matrix = intermediate_coefficients
        test_firms.base_capital_input_use_matrix = np.zeros((n_industries, n_industries))
        test_firms.functions["target_production"].target_inventory_to_demand_fraction = 0.1
        test_firms.functions["target_production"].inventory_adjustment_speed = 1.0
        test_firms.functions["target_production"].financial_constrains_fraction = 0.0
        test_firms.configuration.parameters.firm_activity_finance_revision_mode = "post_credit_cash_budget"
        test_firms.ts.override_current("estimated_demand", np.r_[10.0, np.zeros(n_firms - 1)])
        test_firms.ts.override_current("inventory", np.r_[8.0, np.zeros(n_firms - 1)])
        test_firms.ts.override_current("production", np.r_[100.0, np.zeros(n_firms - 1)])
        test_firms.ts.override_current("target_short_term_credit", np.zeros(n_firms))
        test_firms.ts.override_current("target_long_term_credit", np.zeros(n_firms))
        test_firms.ts.override_current("tfp_multiplier", np.ones(n_firms))
        test_firms.ts.override_current("limiting_intermediate_inputs", np.full(n_firms, 1_000.0))
        test_firms.ts.override_current("limiting_capital_inputs", np.full(n_firms, 1_000.0))
        test_firms.ts.override_current("intermediate_inputs_stock", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("capital_inputs_stock", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("deposits", np.full(n_firms, 1_000.0))
        test_firms.ts.override_current("received_short_term_credit", np.zeros(n_firms))
        test_firms.ts.override_current("received_ordinary_short_term_credit", np.zeros(n_firms))
        test_firms.ts.override_current("received_debt_rollover_credit", np.zeros(n_firms))
        test_firms.ts.override_current("received_overdraft_refinance_credit", np.zeros(n_firms))
        test_firms.ts.override_current("received_long_term_credit", np.zeros(n_firms))
        test_firms.ts.override_current("interest_paid", np.zeros(n_firms))
        test_firms.ts.override_current("debt_installments", np.zeros(n_firms))

        net_target = test_firms.compute_target_production(
            bank_overdraft_rate_on_firm_deposits=np.zeros(
                test_firms.states["Corresponding Bank ID"].max() + 1,
            ),
        )
        test_firms.ts.override_current("target_production", net_target)
        test_firms.ts.override_current("desired_labour_inputs", test_firms.compute_desired_labour_inputs())
        test_firms.ts.override_current(
            "target_intermediate_inputs_production",
            test_firms.compute_target_intermediate_inputs_production(),
        )
        test_firms.ts.override_current(
            "target_capital_inputs_production",
            test_firms.compute_target_capital_inputs_production(),
        )
        unconstrained_intermediate = test_firms.compute_unconstrained_demand_for_intermediate_inputs(
            good_prices=np.ones(n_industries),
        )
        unconstrained_capital = test_firms.compute_unconstrained_demand_for_capital_inputs(
            good_prices=np.ones(n_industries),
        )
        test_firms.ts.override_current("unconstrained_target_intermediate_inputs", unconstrained_intermediate)
        test_firms.ts.override_current(
            "unconstrained_target_intermediate_inputs_costs",
            unconstrained_intermediate.sum(axis=1),
        )
        test_firms.ts.override_current("unconstrained_target_capital_inputs", unconstrained_capital)
        test_firms.ts.override_current("unconstrained_target_capital_inputs_costs", unconstrained_capital.sum(axis=1))
        test_firms.ts.override_current("planned_technical_investment", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("planned_tfp_investment", np.zeros(n_firms))
        test_firms.ts.override_current("planned_productivity_investment", np.zeros(n_firms))

        original_net_target = test_firms.ts.current("target_production").copy()
        assert np.isclose(original_net_target[0], 3.0)

        test_firms.prepare_feasible_activity_plan(
            previous_good_prices=np.ones(n_industries),
            expected_inflation=0.0,
            wage_obligation_preview=np.zeros(n_firms),
            production_tax_obligation_preview=np.zeros(n_firms),
            corporate_tax_obligation_preview=np.zeros(n_firms),
            interest_obligation_preview=np.zeros(n_firms),
            loan_interest_obligation_preview=np.zeros(n_firms),
            debt_installment_preview=np.zeros(n_firms),
        )
        finance_feasible_y = test_firms.ts.current("activity_finance_feasible_target_production").copy()
        assert np.all(finance_feasible_y <= original_net_target + 1e-12)

        test_firms.apply_feasible_labour_demand()
        test_firms.ts.override_current(
            "labour_inputs",
            test_firms.ts.current("activity_finance_feasible_desired_labour_inputs").copy(),
        )
        test_firms.revise_activity_against_realised_labour(expected_lcu_prices=np.ones(n_industries))
        realised_feasible_y = test_firms.ts.current("activity_finance_realised_feasible_target_production").copy()
        assert np.all(realised_feasible_y <= original_net_target + 1e-12)
        assert np.allclose(test_firms.ts.current("target_production"), original_net_target)

        test_firms.prepare_goods_market_orders(
            exchange_rate_usd_to_lcu=1.0,
            previous_good_prices=np.ones(n_industries),
            expected_inflation=0.0,
        )
        assert np.allclose(test_firms.ts.current("target_production"), original_net_target)

    def test__realised_labour_revision_no_rationing_uses_post_credit_feasible_activity(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        n_industries = test_firms.n_industries
        target_intermediate = np.full((n_firms, n_industries), 99.0)
        target_capital = np.full((n_firms, n_industries), 77.0)
        planned_technical = np.full((n_firms, n_industries), 4.0)
        planned_tfp = np.full(n_firms, 5.0)
        target_production = np.full(n_firms, 20.0)
        feasible_y = np.linspace(1.0, float(n_firms), n_firms)
        feasible_labour = np.full(n_firms, 3.0)
        test_firms.base_intermediate_inputs_productivity_matrix = np.ones((n_industries, n_industries))
        test_firms.base_capital_input_use_matrix = np.ones((n_industries, n_industries))
        test_firms.ts.override_current("intermediate_inputs_stock", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("capital_inputs_stock", np.zeros((n_firms, n_industries)))

        test_firms.ts.override_current("target_production", target_production.copy())
        test_firms.ts.override_current("target_intermediate_inputs", target_intermediate.copy())
        test_firms.ts.override_current("target_capital_inputs", target_capital.copy())
        test_firms.ts.override_current("planned_technical_investment", planned_technical.copy())
        test_firms.ts.override_current("planned_tfp_investment", planned_tfp.copy())
        test_firms.ts.override_current("activity_finance_feasible_target_production", feasible_y)
        test_firms.ts.override_current("activity_finance_feasible_desired_labour_inputs", feasible_labour)
        test_firms.ts.override_current("labour_inputs", feasible_labour.copy())

        test_firms.revise_activity_against_realised_labour(expected_lcu_prices=np.ones(n_industries))

        assert np.allclose(test_firms.ts.current("target_production"), target_production)
        assert np.allclose(test_firms.ts.current("target_intermediate_inputs"), feasible_y[:, None])
        assert np.allclose(test_firms.ts.current("target_capital_inputs"), feasible_y[:, None])
        assert np.allclose(test_firms.ts.current("planned_technical_investment"), planned_technical)
        assert np.allclose(test_firms.ts.current("planned_tfp_investment"), planned_tfp)
        assert np.allclose(
            test_firms.ts.current("planned_productivity_investment"),
            planned_tfp + planned_technical.sum(axis=1),
        )
        assert np.allclose(test_firms.ts.current("activity_finance_realised_feasible_target_production"), feasible_y)
        assert np.allclose(test_firms.ts.current("activity_finance_realised_labour_scale"), 1.0)

    def test__realised_labour_revision_rationing_recomputes_activity_goods_only(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        n_industries = test_firms.n_industries
        target_intermediate = np.zeros((n_firms, n_industries))
        target_capital = np.zeros((n_firms, n_industries))
        planned_technical = np.zeros((n_firms, n_industries))
        planned_tfp = np.zeros(n_firms)
        target_intermediate[0, 0] = 20.0
        target_capital[0, 0] = 20.0
        planned_technical[0, 0] = 5.0
        planned_tfp[0] = 5.0

        self._set_simple_activity_solver_inputs(test_firms, target_production=20.0, include_capital=True)
        test_firms.ts.override_current(
            "activity_finance_feasible_target_production", np.r_[20.0, np.zeros(n_firms - 1)]
        )
        test_firms.ts.override_current(
            "activity_finance_feasible_desired_labour_inputs", np.r_[20.0, np.ones(n_firms - 1)]
        )
        test_firms.ts.override_current("labour_inputs", np.r_[10.0, np.ones(n_firms - 1)])
        test_firms.ts.override_current("target_intermediate_inputs", target_intermediate.copy())
        test_firms.ts.override_current("target_capital_inputs", target_capital.copy())
        test_firms.ts.override_current("planned_technical_investment", planned_technical.copy())
        test_firms.ts.override_current("planned_tfp_investment", planned_tfp.copy())

        test_firms.revise_activity_against_realised_labour(expected_lcu_prices=np.ones(n_industries))

        assert np.isclose(test_firms.ts.current("activity_finance_realised_labour_scale")[0], 0.5)
        assert np.isclose(test_firms.ts.current("activity_finance_realised_feasible_target_production")[0], 10.0)
        assert np.isclose(test_firms.ts.current("target_production")[0], 20.0)
        assert np.isclose(test_firms.ts.current("target_intermediate_inputs")[0, 0], 10.0)
        assert np.isclose(test_firms.ts.current("target_capital_inputs")[0, 0], 10.0)
        assert np.isclose(test_firms.ts.current("planned_technical_investment")[0, 0], 5.0)
        assert np.isclose(test_firms.ts.current("planned_tfp_investment")[0], 5.0)
        assert np.isclose(test_firms.ts.current("planned_productivity_investment")[0], 10.0)

    def test__compute_production_uses_realised_feasible_activity_without_overwriting_target(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        n_industries = test_firms.n_industries

        target_production = np.full(n_firms, 20.0)
        feasible_y = np.r_[8.0, np.zeros(n_firms - 1)]
        test_firms.ts.override_current("target_production", target_production.copy())
        test_firms.ts.override_current("activity_finance_feasible_target_production", feasible_y.copy())
        test_firms.ts.override_current("activity_finance_feasible_desired_labour_inputs", feasible_y.copy())
        test_firms.ts.override_current("labour_inputs", np.r_[8.0, np.ones(n_firms - 1)])
        test_firms.ts.override_current("limiting_intermediate_inputs", np.full(n_firms, 100.0))
        test_firms.ts.override_current("limiting_capital_inputs", np.full(n_firms, 100.0))
        test_firms.base_intermediate_inputs_productivity_matrix = np.ones((n_industries, n_industries))
        test_firms.base_capital_input_use_matrix = np.ones((n_industries, n_industries))
        test_firms.ts.override_current("intermediate_inputs_stock", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("capital_inputs_stock", np.zeros((n_firms, n_industries)))

        test_firms.revise_activity_against_realised_labour(expected_lcu_prices=np.ones(n_industries))

        production = test_firms.compute_production()

        assert np.isclose(production[0], 8.0)
        assert np.allclose(test_firms.ts.current("target_production"), target_production)
        assert np.allclose(test_firms.ts.current("activity_finance_realised_feasible_target_production"), feasible_y)

    def test__realised_labour_revision_mixed_firms_resets_all_from_post_credit_feasible(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        n_industries = test_firms.n_industries
        target_production = np.full(n_firms, 50.0)
        feasible_y = np.r_[20.0, 12.0, np.zeros(n_firms - 2)]
        feasible_labour = np.r_[20.0, 12.0, np.ones(n_firms - 2)]
        realised_labour = np.r_[10.0, 12.0, np.ones(n_firms - 2)]
        planned_technical = np.full((n_firms, n_industries), 2.0)
        planned_tfp = np.full(n_firms, 3.0)
        test_firms.base_intermediate_inputs_productivity_matrix = np.ones((n_industries, n_industries))
        test_firms.base_capital_input_use_matrix = np.ones((n_industries, n_industries))
        test_firms.ts.override_current("intermediate_inputs_stock", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("capital_inputs_stock", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("target_production", target_production)
        test_firms.ts.override_current("target_intermediate_inputs", np.full((n_firms, n_industries), 99.0))
        test_firms.ts.override_current("target_capital_inputs", np.full((n_firms, n_industries), 88.0))
        test_firms.ts.override_current("planned_technical_investment", planned_technical.copy())
        test_firms.ts.override_current("planned_tfp_investment", planned_tfp.copy())
        test_firms.ts.override_current("activity_finance_feasible_target_production", feasible_y)
        test_firms.ts.override_current("activity_finance_feasible_desired_labour_inputs", feasible_labour)
        test_firms.ts.override_current("labour_inputs", realised_labour)

        test_firms.revise_activity_against_realised_labour(expected_lcu_prices=np.ones(n_industries))

        expected_y = np.r_[10.0, 12.0, np.zeros(n_firms - 2)]
        assert np.allclose(test_firms.ts.current("target_production"), target_production)
        assert np.allclose(test_firms.ts.current("target_intermediate_inputs"), expected_y[:, None])
        assert np.allclose(test_firms.ts.current("target_capital_inputs"), expected_y[:, None])
        assert np.allclose(test_firms.ts.current("planned_technical_investment"), planned_technical)
        assert np.allclose(test_firms.ts.current("planned_tfp_investment"), planned_tfp)
        assert np.allclose(test_firms.ts.current("activity_finance_realised_labour_scale")[:2], [0.5, 1.0])

    def test__realised_labour_revision_zero_feasible_labour_uses_unit_scale(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        n_industries = test_firms.n_industries
        target_production = np.full(n_firms, 7.0)

        test_firms.ts.override_current("target_production", target_production.copy())
        test_firms.ts.override_current("activity_finance_feasible_target_production", target_production.copy())
        test_firms.ts.override_current("activity_finance_feasible_desired_labour_inputs", np.zeros(n_firms))
        test_firms.ts.override_current("labour_inputs", np.zeros(n_firms))

        test_firms.revise_activity_against_realised_labour(expected_lcu_prices=np.ones(n_industries))

        assert np.allclose(test_firms.ts.current("activity_finance_realised_labour_scale"), 1.0)
        assert np.allclose(test_firms.ts.current("target_production"), target_production)

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

    def test__compute_target_credit_uses_pro_forma_cash_budget(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        n_industries = test_firms.n_industries

        test_firms.ts.override_current("deposits", np.r_[10.0, np.zeros(n_firms - 1)])
        test_firms.ts.override_current("price", np.r_[2.0, np.ones(n_firms - 1)])
        test_firms.ts.override_current("target_production", np.r_[10.0, np.zeros(n_firms - 1)])
        test_firms.ts.override_current("corporate_taxes_paid", np.zeros(n_firms))
        test_firms.ts.override_current("interest_paid", np.r_[4.0, np.zeros(n_firms - 1)])
        test_firms.ts.override_current("debt_installments", np.r_[1.0, np.zeros(n_firms - 1)])
        test_firms.ts.override_current(
            "unconstrained_target_intermediate_inputs_costs",
            np.r_[12.0, np.zeros(n_firms - 1)],
        )
        test_firms.ts.override_current(
            "unconstrained_target_capital_inputs_costs",
            np.r_[8.0, np.zeros(n_firms - 1)],
        )
        planned_technical = np.zeros((n_firms, n_industries))
        planned_technical[0, :2] = [1.5, 0.5]
        test_firms.ts.override_current("planned_technical_investment", planned_technical)
        test_firms.ts.override_current("planned_tfp_investment", np.r_[3.0, np.zeros(n_firms - 1)])

        test_firms.compute_target_credit(
            estimated_growth=0.0,
            estimated_inflation=0.0,
            wage_obligation_preview=np.r_[15.0, np.zeros(n_firms - 1)],
            production_tax_obligation_preview=np.r_[5.0, np.zeros(n_firms - 1)],
        )

        assert np.isclose(test_firms.ts.current("target_short_term_credit")[0], 15.0)
        assert np.isclose(test_firms.ts.current("target_debt_rollover_credit")[0], 0.0)
        assert np.isclose(test_firms.ts.current("target_overdraft_refinance_credit")[0], 0.0)
        assert np.isclose(test_firms.ts.current("ordinary_target_short_term_credit")[0], 15.0)
        assert np.isclose(test_firms.ts.current("target_long_term_credit")[0], 10.0)
        assert np.isclose(test_firms.ts.current("credit_budget_internal_cash")[0], 10.0)
        assert np.isclose(test_firms.ts.current("credit_budget_existing_overdraft")[0], 0.0)
        assert np.isclose(test_firms.ts.current("expected_sales")[0], 20.0)
        assert np.isclose(test_firms.ts.current("credit_budget_wage_obligations")[0], 15.0)
        assert np.isclose(test_firms.ts.current("credit_budget_production_tax_obligations")[0], 5.0)
        assert np.isclose(test_firms.ts.current("credit_budget_corporate_tax_obligations")[0], 0.0)
        assert np.isclose(test_firms.ts.current("credit_budget_interest_obligations")[0], 4.0)
        assert np.isclose(test_firms.ts.current("credit_budget_debt_installments")[0], 1.0)
        assert np.isclose(test_firms.ts.current("credit_budget_hard_obligations")[0], 25.0)
        assert np.isclose(test_firms.ts.current("credit_budget_cash_after_hard_obligations")[0], 5.0)
        assert np.isclose(test_firms.ts.current("credit_budget_available_after_hard_and_overdraft")[0], 5.0)
        assert np.isclose(test_firms.ts.current("credit_budget_intermediate_costs")[0], 12.0)
        assert np.isclose(test_firms.ts.current("credit_budget_tfp_costs")[0], 3.0)
        assert np.isclose(test_firms.ts.current("credit_budget_working_capital_budget")[0], 15.0)
        assert np.isclose(test_firms.ts.current("credit_budget_capital_costs")[0], 8.0)
        assert np.isclose(test_firms.ts.current("credit_budget_technical_investment_costs")[0], 2.0)
        assert np.isclose(test_firms.ts.current("credit_budget_investment_budget")[0], 10.0)
        assert np.isclose(
            test_firms.ts.current("credit_budget_remaining_internal_finance_after_working_capital")[0],
            -10.0,
        )
        assert np.allclose(test_firms.ts.current("target_short_term_credit")[1:], 0.0)
        assert np.allclose(test_firms.ts.current("target_long_term_credit")[1:], 0.0)

        cash_after_hard = test_firms.ts.current("credit_budget_cash_after_hard_obligations")
        assert np.allclose(
            cash_after_hard,
            test_firms.ts.current("credit_budget_internal_cash")
            + test_firms.ts.current("expected_sales")
            - test_firms.ts.current("credit_budget_hard_obligations"),
        )
        assert np.allclose(
            test_firms.ts.current("credit_budget_available_after_hard_and_overdraft"),
            cash_after_hard - test_firms.ts.current("credit_budget_existing_overdraft"),
        )
        assert np.allclose(
            test_firms.ts.current("credit_budget_working_capital_budget"),
            test_firms.ts.current("credit_budget_intermediate_costs")
            + test_firms.ts.current("credit_budget_tfp_costs"),
        )

    def test__compute_target_credit_refinances_existing_overdraft_as_short_term_credit(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        n_industries = test_firms.n_industries

        test_firms.ts.override_current("deposits", np.r_[-100.0, np.zeros(n_firms - 1)])
        test_firms.ts.override_current("price", np.ones(n_firms))
        test_firms.ts.override_current("target_production", np.zeros(n_firms))
        test_firms.ts.override_current("corporate_taxes_paid", np.zeros(n_firms))
        test_firms.ts.override_current("interest_paid", np.zeros(n_firms))
        test_firms.ts.override_current("debt_installments", np.zeros(n_firms))
        test_firms.ts.override_current("unconstrained_target_intermediate_inputs_costs", np.zeros(n_firms))
        test_firms.ts.override_current("unconstrained_target_capital_inputs_costs", np.zeros(n_firms))
        test_firms.ts.override_current("planned_technical_investment", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("planned_tfp_investment", np.zeros(n_firms))

        test_firms.compute_target_credit(
            estimated_growth=0.0,
            estimated_inflation=0.0,
            wage_obligation_preview=np.zeros(n_firms),
            production_tax_obligation_preview=np.zeros(n_firms),
        )

        assert np.isclose(test_firms.ts.current("target_short_term_credit")[0], 100.0)
        assert np.isclose(test_firms.ts.current("target_debt_rollover_credit")[0], 0.0)
        assert np.isclose(test_firms.ts.current("target_overdraft_refinance_credit")[0], 100.0)
        assert np.isclose(test_firms.ts.current("ordinary_target_short_term_credit")[0], 0.0)
        assert np.isclose(test_firms.ts.current("target_long_term_credit")[0], 0.0)
        assert np.isclose(test_firms.ts.current("credit_budget_internal_cash")[0], 0.0)
        assert np.isclose(test_firms.ts.current("credit_budget_existing_overdraft")[0], 100.0)
        assert np.isclose(test_firms.ts.current("expected_sales")[0], 0.0)
        assert np.isclose(test_firms.ts.current("credit_budget_hard_obligations")[0], 0.0)
        assert np.isclose(test_firms.ts.current("credit_budget_cash_after_hard_obligations")[0], 0.0)
        assert np.isclose(test_firms.ts.current("credit_budget_available_after_hard_and_overdraft")[0], -100.0)
        assert np.isclose(test_firms.ts.current("credit_budget_working_capital_budget")[0], 0.0)
        assert np.isclose(
            test_firms.ts.current("credit_budget_remaining_internal_finance_after_working_capital")[0],
            -100.0,
        )
        assert np.allclose(
            test_firms.ts.current("credit_budget_available_after_hard_and_overdraft"),
            test_firms.ts.current("credit_budget_cash_after_hard_obligations")
            - test_firms.ts.current("credit_budget_existing_overdraft"),
        )
        assert np.allclose(test_firms.ts.current("target_short_term_credit")[1:], 0.0)
        assert np.allclose(test_firms.ts.current("target_long_term_credit")[1:], 0.0)

    def test__compute_target_credit_splits_scheduled_principal_rollover_from_ordinary_st(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        n_industries = test_firms.n_industries

        test_firms.ts.override_current("deposits", np.r_[10.0, np.zeros(n_firms - 1)])
        test_firms.ts.override_current("price", np.ones(n_firms))
        test_firms.ts.override_current("target_production", np.zeros(n_firms))
        test_firms.ts.override_current("corporate_taxes_paid", np.zeros(n_firms))
        test_firms.ts.override_current("interest_paid", np.r_[2.0, np.zeros(n_firms - 1)])
        test_firms.ts.override_current("debt_installments", np.r_[100.0, np.zeros(n_firms - 1)])
        test_firms.ts.override_current("unconstrained_target_intermediate_inputs_costs", np.zeros(n_firms))
        test_firms.ts.override_current("unconstrained_target_capital_inputs_costs", np.zeros(n_firms))
        test_firms.ts.override_current("planned_technical_investment", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("planned_tfp_investment", np.zeros(n_firms))

        test_firms.compute_target_credit(
            estimated_growth=0.0,
            estimated_inflation=0.0,
            wage_obligation_preview=np.r_[3.0, np.zeros(n_firms - 1)],
            production_tax_obligation_preview=np.zeros(n_firms),
        )

        assert np.isclose(test_firms.ts.current("target_short_term_credit")[0], 95.0)
        assert np.isclose(test_firms.ts.current("target_debt_rollover_credit")[0], 95.0)
        assert np.isclose(test_firms.ts.current("target_overdraft_refinance_credit")[0], 0.0)
        assert np.isclose(test_firms.ts.current("ordinary_target_short_term_credit")[0], 0.0)
        assert np.isclose(test_firms.ts.current("credit_budget_cash_after_hard_obligations")[0], -95.0)
        assert np.isclose(test_firms.ts.current("credit_budget_working_capital_budget")[0], 0.0)

    def test__compute_target_credit_uses_current_service_previews(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        n_industries = test_firms.n_industries

        test_firms.ts.override_current("deposits", np.r_[10.0, np.zeros(n_firms - 1)])
        test_firms.ts.override_current("price", np.ones(n_firms))
        test_firms.ts.override_current("target_production", np.zeros(n_firms))
        test_firms.ts.override_current("corporate_taxes_paid", np.zeros(n_firms))
        test_firms.ts.override_current("interest_paid", np.r_[99.0, np.zeros(n_firms - 1)])
        test_firms.ts.override_current("debt_installments", np.r_[99.0, np.zeros(n_firms - 1)])
        test_firms.ts.override_current("unconstrained_target_intermediate_inputs_costs", np.zeros(n_firms))
        test_firms.ts.override_current("unconstrained_target_capital_inputs_costs", np.zeros(n_firms))
        test_firms.ts.override_current("planned_technical_investment", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("planned_tfp_investment", np.zeros(n_firms))

        test_firms.compute_target_credit(
            estimated_growth=0.0,
            estimated_inflation=0.0,
            wage_obligation_preview=np.r_[3.0, np.zeros(n_firms - 1)],
            production_tax_obligation_preview=np.zeros(n_firms),
            interest_obligation_preview=np.r_[2.0, np.zeros(n_firms - 1)],
            debt_installment_preview=np.r_[8.0, np.zeros(n_firms - 1)],
        )

        assert np.isclose(test_firms.ts.current("credit_budget_interest_obligations")[0], 2.0)
        assert np.isclose(test_firms.ts.current("credit_budget_debt_installments")[0], 8.0)
        assert np.isclose(test_firms.ts.current("target_debt_rollover_credit")[0], 3.0)
        assert np.isclose(test_firms.ts.current("ordinary_target_short_term_credit")[0], 0.0)

    def test__compute_target_credit_rollover_covers_unfunded_loan_interest_and_principal(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        n_industries = test_firms.n_industries

        test_firms.ts.override_current("deposits", np.r_[10.0, np.zeros(n_firms - 1)])
        test_firms.ts.override_current("price", np.ones(n_firms))
        test_firms.ts.override_current("target_production", np.zeros(n_firms))
        test_firms.ts.override_current("corporate_taxes_paid", np.zeros(n_firms))
        test_firms.ts.override_current("interest_paid", np.zeros(n_firms))
        test_firms.ts.override_current("debt_installments", np.zeros(n_firms))
        test_firms.ts.override_current("unconstrained_target_intermediate_inputs_costs", np.zeros(n_firms))
        test_firms.ts.override_current("unconstrained_target_capital_inputs_costs", np.zeros(n_firms))
        test_firms.ts.override_current("planned_technical_investment", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("planned_tfp_investment", np.zeros(n_firms))

        test_firms.compute_target_credit(
            estimated_growth=0.0,
            estimated_inflation=0.0,
            wage_obligation_preview=np.r_[7.0, np.zeros(n_firms - 1)],
            production_tax_obligation_preview=np.zeros(n_firms),
            interest_obligation_preview=np.r_[5.0, np.zeros(n_firms - 1)],
            loan_interest_obligation_preview=np.r_[5.0, np.zeros(n_firms - 1)],
            debt_installment_preview=np.r_[8.0, np.zeros(n_firms - 1)],
        )

        assert np.isclose(test_firms.ts.current("target_debt_rollover_credit")[0], 10.0)
        assert np.isclose(test_firms.ts.current("ordinary_target_short_term_credit")[0], 0.0)

    def test__compute_target_credit_includes_non_principal_hard_shortfall_in_ordinary_st(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        n_industries = test_firms.n_industries

        test_firms.ts.override_current("deposits", np.zeros(n_firms))
        test_firms.ts.override_current("price", np.ones(n_firms))
        test_firms.ts.override_current("target_production", np.zeros(n_firms))
        test_firms.ts.override_current("corporate_taxes_paid", np.zeros(n_firms))
        test_firms.ts.override_current("interest_paid", np.zeros(n_firms))
        test_firms.ts.override_current("debt_installments", np.zeros(n_firms))
        test_firms.ts.override_current("unconstrained_target_intermediate_inputs_costs", np.zeros(n_firms))
        test_firms.ts.override_current("unconstrained_target_capital_inputs_costs", np.zeros(n_firms))
        test_firms.ts.override_current("planned_technical_investment", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("planned_tfp_investment", np.zeros(n_firms))

        test_firms.compute_target_credit(
            estimated_growth=0.0,
            estimated_inflation=0.0,
            wage_obligation_preview=np.r_[5.0, np.zeros(n_firms - 1)],
            production_tax_obligation_preview=np.r_[2.0, np.zeros(n_firms - 1)],
            interest_obligation_preview=np.r_[3.0, np.zeros(n_firms - 1)],
            debt_installment_preview=np.zeros(n_firms),
        )

        assert np.isclose(test_firms.ts.current("target_short_term_credit")[0], 10.0)
        assert np.isclose(test_firms.ts.current("ordinary_target_short_term_credit")[0], 10.0)
        assert np.isclose(test_firms.ts.current("target_debt_rollover_credit")[0], 0.0)
        assert np.isclose(test_firms.ts.current("target_overdraft_refinance_credit")[0], 0.0)
        assert np.isclose(test_firms.ts.current("target_long_term_credit")[0], 0.0)

    def test__plan_firm_debt_settlement_keeps_rollover_and_refinance_ring_fenced(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        n_industries = test_firms.n_industries

        test_firms.begin_firm_debt_settlement(
            opening_interest_arrears=np.zeros(n_firms),
            opening_principal_arrears=np.zeros(n_firms),
            contractual_interest_due=np.r_[4.0, np.zeros(n_firms - 1)],
            contractual_principal_due=np.r_[10.0, np.zeros(n_firms - 1)],
            scheduled_interest_due=np.r_[4.0, np.zeros(n_firms - 1)],
            scheduled_principal_due=np.r_[10.0, np.zeros(n_firms - 1)],
        )
        test_firms.ts.override_current("deposits", np.r_[-10.0, np.zeros(n_firms - 1)])
        test_firms.ts.override_current("nominal_amount_sold_in_lcu", np.r_[6.0, np.zeros(n_firms - 1)])
        test_firms.ts.override_current("total_wage", np.zeros(n_firms))
        test_firms.ts.override_current("nominal_amount_spent_in_lcu", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("received_short_term_credit", np.r_[13.0, np.zeros(n_firms - 1)])
        test_firms.ts.override_current("received_debt_rollover_credit", np.r_[5.0, np.zeros(n_firms - 1)])
        test_firms.ts.override_current("received_overdraft_refinance_credit", np.r_[8.0, np.zeros(n_firms - 1)])
        test_firms.ts.override_current("received_ordinary_short_term_credit", np.zeros(n_firms))
        test_firms.ts.override_current("received_long_term_credit", np.zeros(n_firms))
        test_firms.ts.override_current("received_credit", np.r_[13.0, np.zeros(n_firms - 1)])
        test_firms.ts.override_current("target_debt_rollover_credit", np.r_[10.0, np.zeros(n_firms - 1)])
        test_firms.ts.override_current("target_overdraft_refinance_credit", np.r_[10.0, np.zeros(n_firms - 1)])
        test_firms.ts.override_current("planned_technical_investment", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("planned_tfp_investment", np.zeros(n_firms))
        test_firms.ts.override_current("direct_tfp_investment_cash_expense", np.zeros(n_firms))

        settlement = test_firms.plan_firm_debt_settlement(
            taxes_paid_on_production=np.zeros(n_firms),
            interest_paid_on_deposits=np.zeros(n_firms),
            corporate_tax_obligation_preview=np.zeros(n_firms),
        )

        assert np.isclose(settlement["available_cash_before_debt_service"][0], -4.0)
        assert np.isclose(settlement["corporate_tax_reserve"][0], 0.0)
        assert np.isclose(settlement["cash_after_tax_reserve"][0], -4.0)
        assert np.isclose(settlement["payable_interest"][0], 4.0)
        assert np.isclose(settlement["payable_opening_interest_arrears"][0], 0.0)
        assert np.isclose(settlement["payable_contractual_interest"][0], 4.0)
        assert np.isclose(settlement["payable_principal"][0], 1.0)
        assert np.isclose(settlement["payable_opening_principal_arrears"][0], 0.0)
        assert np.isclose(settlement["payable_contractual_principal"][0], 1.0)
        assert np.isclose(settlement["capitalized_interest"][0], 0.0)
        assert np.isclose(settlement["closing_interest_arrears"][0], 0.0)
        assert np.isclose(settlement["closing_principal_arrears"][0], 9.0)
        assert np.isclose(settlement["debt_rollover_shortfall"][0], 5.0)
        assert np.isclose(settlement["overdraft_refinance_used"][0], 4.0)
        assert np.isclose(settlement["overdraft_refinance_shortfall"][0], 2.0)
        assert settlement["illiquid_flag"][0]
        assert not settlement["default_flag"][0]

    def test__plan_firm_debt_settlement_does_not_use_rollover_to_skip_cash_hole(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        n_industries = test_firms.n_industries

        test_firms.begin_firm_debt_settlement(
            opening_interest_arrears=np.zeros(n_firms),
            opening_principal_arrears=np.zeros(n_firms),
            contractual_interest_due=np.r_[4.0, np.zeros(n_firms - 1)],
            contractual_principal_due=np.r_[10.0, np.zeros(n_firms - 1)],
            scheduled_interest_due=np.r_[4.0, np.zeros(n_firms - 1)],
            scheduled_principal_due=np.r_[10.0, np.zeros(n_firms - 1)],
        )
        test_firms.ts.override_current("deposits", np.r_[-10.0, np.zeros(n_firms - 1)])
        test_firms.ts.override_current("nominal_amount_sold_in_lcu", np.r_[6.0, np.zeros(n_firms - 1)])
        test_firms.ts.override_current("total_wage", np.zeros(n_firms))
        test_firms.ts.override_current("nominal_amount_spent_in_lcu", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("received_short_term_credit", np.r_[5.0, np.zeros(n_firms - 1)])
        test_firms.ts.override_current("received_debt_rollover_credit", np.r_[5.0, np.zeros(n_firms - 1)])
        test_firms.ts.override_current("received_overdraft_refinance_credit", np.zeros(n_firms))
        test_firms.ts.override_current("received_ordinary_short_term_credit", np.zeros(n_firms))
        test_firms.ts.override_current("received_long_term_credit", np.zeros(n_firms))
        test_firms.ts.override_current("target_debt_rollover_credit", np.r_[10.0, np.zeros(n_firms - 1)])
        test_firms.ts.override_current("target_overdraft_refinance_credit", np.zeros(n_firms))
        test_firms.ts.override_current("planned_technical_investment", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("planned_tfp_investment", np.zeros(n_firms))
        test_firms.ts.override_current("direct_tfp_investment_cash_expense", np.zeros(n_firms))

        settlement = test_firms.plan_firm_debt_settlement(
            taxes_paid_on_production=np.zeros(n_firms),
            interest_paid_on_deposits=np.zeros(n_firms),
            corporate_tax_obligation_preview=np.zeros(n_firms),
        )

        assert np.isclose(settlement["cash_after_tax_reserve"][0], -4.0)
        assert np.isclose(settlement["payable_interest"][0], 1.0)
        assert np.isclose(settlement["capitalized_interest"][0], 3.0)
        assert np.isclose(settlement["payable_principal"][0], 0.0)
        assert np.isclose(settlement["closing_principal_arrears"][0], 10.0)
        assert settlement["illiquid_flag"][0]

    def test__plan_firm_debt_settlement_capitalizes_interest_and_flags_illiquidity(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        n_industries = test_firms.n_industries

        test_firms.begin_firm_debt_settlement(
            opening_interest_arrears=np.r_[1.0, np.zeros(n_firms - 1)],
            opening_principal_arrears=np.r_[2.0, np.zeros(n_firms - 1)],
            contractual_interest_due=np.r_[3.0, np.zeros(n_firms - 1)],
            contractual_principal_due=np.r_[4.0, np.zeros(n_firms - 1)],
            scheduled_interest_due=np.r_[4.0, np.zeros(n_firms - 1)],
            scheduled_principal_due=np.r_[6.0, np.zeros(n_firms - 1)],
        )
        test_firms.ts.override_current("deposits", np.zeros(n_firms))
        test_firms.ts.override_current("nominal_amount_sold_in_lcu", np.zeros(n_firms))
        test_firms.ts.override_current("total_wage", np.zeros(n_firms))
        test_firms.ts.override_current("nominal_amount_spent_in_lcu", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("received_short_term_credit", np.zeros(n_firms))
        test_firms.ts.override_current("received_debt_rollover_credit", np.zeros(n_firms))
        test_firms.ts.override_current("received_overdraft_refinance_credit", np.zeros(n_firms))
        test_firms.ts.override_current("received_ordinary_short_term_credit", np.zeros(n_firms))
        test_firms.ts.override_current("received_long_term_credit", np.zeros(n_firms))
        test_firms.ts.override_current("received_credit", np.zeros(n_firms))
        test_firms.ts.override_current("target_debt_rollover_credit", np.zeros(n_firms))
        test_firms.ts.override_current("target_overdraft_refinance_credit", np.zeros(n_firms))
        test_firms.ts.override_current("planned_technical_investment", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("planned_tfp_investment", np.zeros(n_firms))
        test_firms.ts.override_current("direct_tfp_investment_cash_expense", np.zeros(n_firms))

        settlement = test_firms.plan_firm_debt_settlement(
            taxes_paid_on_production=np.zeros(n_firms),
            interest_paid_on_deposits=np.zeros(n_firms),
            corporate_tax_obligation_preview=np.zeros(n_firms),
        )

        assert np.isclose(settlement["capitalized_interest"][0], 4.0)
        assert np.isclose(settlement["closing_interest_arrears"][0], 0.0)
        assert np.isclose(settlement["closing_principal_arrears"][0], 6.0)
        assert settlement["illiquid_flag"][0]
        assert not settlement["default_flag"][0]

    def test__plan_firm_debt_settlement_grants_new_grace_if_opening_arrears_are_cured(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        n_industries = test_firms.n_industries

        test_firms.begin_firm_debt_settlement(
            opening_interest_arrears=np.r_[1.0, np.zeros(n_firms - 1)],
            opening_principal_arrears=np.r_[2.0, np.zeros(n_firms - 1)],
            contractual_interest_due=np.r_[3.0, np.zeros(n_firms - 1)],
            contractual_principal_due=np.r_[4.0, np.zeros(n_firms - 1)],
            scheduled_interest_due=np.r_[4.0, np.zeros(n_firms - 1)],
            scheduled_principal_due=np.r_[6.0, np.zeros(n_firms - 1)],
        )
        test_firms.ts.override_current("deposits", np.r_[6.0, np.zeros(n_firms - 1)])
        test_firms.ts.override_current("nominal_amount_sold_in_lcu", np.zeros(n_firms))
        test_firms.ts.override_current("total_wage", np.zeros(n_firms))
        test_firms.ts.override_current("nominal_amount_spent_in_lcu", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("received_short_term_credit", np.zeros(n_firms))
        test_firms.ts.override_current("received_debt_rollover_credit", np.zeros(n_firms))
        test_firms.ts.override_current("received_overdraft_refinance_credit", np.zeros(n_firms))
        test_firms.ts.override_current("received_ordinary_short_term_credit", np.zeros(n_firms))
        test_firms.ts.override_current("received_long_term_credit", np.zeros(n_firms))
        test_firms.ts.override_current("received_credit", np.zeros(n_firms))
        test_firms.ts.override_current("target_debt_rollover_credit", np.zeros(n_firms))
        test_firms.ts.override_current("target_overdraft_refinance_credit", np.zeros(n_firms))
        test_firms.ts.override_current("planned_technical_investment", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("planned_tfp_investment", np.zeros(n_firms))
        test_firms.ts.override_current("direct_tfp_investment_cash_expense", np.zeros(n_firms))

        settlement = test_firms.plan_firm_debt_settlement(
            taxes_paid_on_production=np.zeros(n_firms),
            interest_paid_on_deposits=np.zeros(n_firms),
            corporate_tax_obligation_preview=np.zeros(n_firms),
        )

        assert np.isclose(settlement["payable_opening_interest_arrears"][0], 1.0)
        assert np.isclose(settlement["payable_opening_principal_arrears"][0], 2.0)
        assert np.isclose(settlement["capitalized_interest"][0], 0.0)
        assert np.isclose(settlement["closing_interest_arrears"][0], 0.0)
        assert np.isclose(settlement["closing_principal_arrears"][0], 4.0)
        assert settlement["illiquid_flag"][0]
        assert not settlement["default_flag"][0]

    def test__plan_firm_debt_settlement_reserves_preview_corporate_tax_before_debt_service(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        n_industries = test_firms.n_industries

        test_firms.begin_firm_debt_settlement(
            opening_interest_arrears=np.zeros(n_firms),
            opening_principal_arrears=np.zeros(n_firms),
            contractual_interest_due=np.r_[4.0, np.zeros(n_firms - 1)],
            contractual_principal_due=np.r_[6.0, np.zeros(n_firms - 1)],
            scheduled_interest_due=np.r_[4.0, np.zeros(n_firms - 1)],
            scheduled_principal_due=np.r_[6.0, np.zeros(n_firms - 1)],
        )
        test_firms.ts.override_current("deposits", np.r_[10.0, np.zeros(n_firms - 1)])
        test_firms.ts.override_current("nominal_amount_sold_in_lcu", np.zeros(n_firms))
        test_firms.ts.override_current("total_wage", np.zeros(n_firms))
        test_firms.ts.override_current("nominal_amount_spent_in_lcu", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("received_short_term_credit", np.zeros(n_firms))
        test_firms.ts.override_current("received_debt_rollover_credit", np.zeros(n_firms))
        test_firms.ts.override_current("received_overdraft_refinance_credit", np.zeros(n_firms))
        test_firms.ts.override_current("received_ordinary_short_term_credit", np.zeros(n_firms))
        test_firms.ts.override_current("received_long_term_credit", np.zeros(n_firms))
        test_firms.ts.override_current("received_credit", np.zeros(n_firms))
        test_firms.ts.override_current("target_debt_rollover_credit", np.zeros(n_firms))
        test_firms.ts.override_current("target_overdraft_refinance_credit", np.zeros(n_firms))
        test_firms.ts.override_current("planned_technical_investment", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("planned_tfp_investment", np.zeros(n_firms))
        test_firms.ts.override_current("direct_tfp_investment_cash_expense", np.zeros(n_firms))

        settlement = test_firms.plan_firm_debt_settlement(
            taxes_paid_on_production=np.zeros(n_firms),
            interest_paid_on_deposits=np.zeros(n_firms),
            corporate_tax_obligation_preview=np.r_[7.0, np.zeros(n_firms - 1)],
        )

        assert np.isclose(settlement["available_cash_before_debt_service"][0], 10.0)
        assert np.isclose(settlement["corporate_tax_reserve"][0], 7.0)
        assert np.isclose(settlement["cash_after_tax_reserve"][0], 3.0)
        assert np.isclose(settlement["payable_interest"][0], 3.0)
        assert np.isclose(settlement["capitalized_interest"][0], 1.0)
        assert np.isclose(settlement["payable_principal"][0], 0.0)

    def test__finalize_firm_debt_settlement_keeps_unpaid_aliases_in_sync(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        settlement = {
            "available_cash_before_debt_service": np.r_[5.0, np.zeros(n_firms - 1)],
            "corporate_tax_reserve": np.r_[1.0, np.zeros(n_firms - 1)],
            "cash_after_tax_reserve": np.r_[4.0, np.zeros(n_firms - 1)],
            "payable_interest": np.r_[2.0, np.zeros(n_firms - 1)],
            "closing_interest_arrears": np.zeros(n_firms),
            "capitalized_interest": np.r_[3.0, np.zeros(n_firms - 1)],
            "payable_principal": np.r_[4.0, np.zeros(n_firms - 1)],
            "closing_principal_arrears": np.r_[6.0, np.zeros(n_firms - 1)],
            "illiquid_flag": np.r_[True, np.full(n_firms - 1, False)],
            "debt_rollover_shortfall": np.zeros(n_firms),
            "overdraft_refinance_used": np.r_[2.0, np.zeros(n_firms - 1)],
            "overdraft_refinance_shortfall": np.zeros(n_firms),
            "default_flag": np.r_[True, np.full(n_firms - 1, False)],
        }

        test_firms.begin_firm_debt_settlement(
            opening_interest_arrears=np.zeros(n_firms),
            opening_principal_arrears=np.zeros(n_firms),
            contractual_interest_due=np.zeros(n_firms),
            contractual_principal_due=np.zeros(n_firms),
            scheduled_interest_due=np.zeros(n_firms),
            scheduled_principal_due=np.zeros(n_firms),
        )
        test_firms.finalize_firm_debt_settlement(
            settlement=settlement,
            residual_overdraft_exposure=np.r_[7.0, np.zeros(n_firms - 1)],
        )

        assert np.isclose(test_firms.ts.current("firm_settlement_closing_interest_arrears")[0], 0.0)
        assert np.isclose(test_firms.ts.current("firm_settlement_unpaid_interest")[0], 3.0)
        assert np.isclose(test_firms.ts.current("firm_settlement_capitalized_interest")[0], 3.0)
        assert np.isclose(test_firms.ts.current("firm_settlement_closing_principal_arrears")[0], 6.0)
        assert np.isclose(test_firms.ts.current("firm_settlement_unpaid_principal")[0], 6.0)
        assert np.isclose(test_firms.ts.current("firm_settlement_overdraft_refinance_used")[0], 2.0)
        assert test_firms.ts.current("firm_settlement_illiquid_flag")[0]
        assert np.isclose(test_firms.ts.current("firm_settlement_residual_overdraft_exposure")[0], 7.0)

    def test__compute_total_credit_exposure_includes_negative_deposits(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        test_firms.ts.override_current("short_term_loan_debt", np.r_[3.0, np.zeros(n_firms - 1)])
        test_firms.ts.override_current("long_term_loan_debt", np.r_[10.0, np.zeros(n_firms - 1)])
        test_firms.ts.override_current("deposits", np.r_[-7.0, np.zeros(n_firms - 1)])

        exposure = test_firms.compute_total_credit_exposure()

        assert np.isclose(exposure[0], 20.0)

    def test__capital_depreciation_matrix_alias_updates_capital_input_use_matrix(self, test_firms):
        n_industries = test_firms.n_industries
        replacement_coefficients = np.eye(n_industries)

        test_firms.base_capital_inputs_depreciation_matrix = replacement_coefficients

        assert np.allclose(test_firms.base_capital_input_use_matrix, replacement_coefficients)
        assert np.allclose(test_firms.base_capital_inputs_depreciation_matrix, replacement_coefficients)

    def test__current_goods_criticality_by_firm_falls_back_for_legacy_18_sector_matrix(self, test_firms):
        original_n_industries = test_firms.n_industries
        test_firms.n_industries = 43
        test_firms.goods_criticality_matrix = np.eye(18)

        try:
            criticality = test_firms.current_goods_criticality_by_firm()
        finally:
            test_firms.n_industries = original_n_industries

        assert criticality.shape == (test_firms.ts.current("n_firms"), 43)
        assert np.allclose(criticality, 1.0)

    def test__handle_insolvency_does_not_reset_balance_sheet_insolvent_liquid_firm(
        self,
        test_firms,
        test_credit_market,
    ):
        n_firms = test_firms.ts.current("n_firms")
        test_firms.ts.override_current("equity", np.r_[-1.0, np.full(n_firms - 1, 10.0)])
        test_firms.ts.override_current("deposits", np.full(n_firms, 10.0))

        test_firms.handle_insolvency(credit_market=test_credit_market, illiquid_flag=np.full(n_firms, False))

        assert not test_firms.states["is_insolvent"][0]
        assert np.isclose(test_firms.ts.current("equity")[0], -1.0)
        assert np.isclose(test_firms.ts.current("deposits")[0], 10.0)

    def test__handle_insolvency_does_not_reset_balance_sheet_solvent_illiquid_firm(
        self, test_firms, test_credit_market
    ):
        n_firms = test_firms.ts.current("n_firms")
        illiquid_flag = np.full(n_firms, False)
        illiquid_flag[0] = True
        test_firms.ts.override_current("equity", np.full(n_firms, 10.0))
        test_firms.ts.override_current("deposits", np.full(n_firms, 10.0))

        test_firms.handle_insolvency(credit_market=test_credit_market, illiquid_flag=illiquid_flag)

        assert not test_firms.states["is_insolvent"][0]
        assert np.isclose(test_firms.ts.current("equity")[0], 10.0)
        assert np.isclose(test_firms.ts.current("deposits")[0], 10.0)

    def test__handle_insolvency_resets_only_insolvent_illiquid_firm(self, test_firms, test_credit_market):
        n_firms = test_firms.ts.current("n_firms")
        illiquid_flag = np.full(n_firms, False)
        illiquid_flag[0] = True
        test_firms.ts.override_current("equity", np.r_[-1.0, np.full(n_firms - 1, 10.0)])
        test_firms.ts.override_current("deposits", np.full(n_firms, 10.0))

        test_firms.handle_insolvency(credit_market=test_credit_market, illiquid_flag=illiquid_flag)

        assert test_firms.states["is_insolvent"][0]
        assert np.isclose(test_firms.ts.current("equity")[0], 0.0)
        assert np.isclose(test_firms.ts.current("deposits")[0], 0.0)

    def test__handle_insolvency_returns_current_default_bank_credit_losses(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        n_banks = 2
        st_loans = np.zeros((3, n_banks, n_firms))
        lt_loans = np.zeros((3, n_banks, n_firms))
        st_loans[0, 0, 0] = 100.0
        lt_loans[0, 1, 0] = 25.0
        st_loans[0, 0, 1] = 50.0
        credit_market = CreditMarket.from_data(
            country_name="TST",
            st_loans=st_loans,
            lt_loans=lt_loans,
            cons_loans=np.zeros((3, n_banks, 1)),
            mort_loans=np.zeros((3, n_banks, 1)),
        )
        corresponding_bank = test_firms.states["Corresponding Bank ID"].copy()
        corresponding_bank[:2] = np.array([1, 0])
        test_firms.states["Corresponding Bank ID"] = corresponding_bank
        test_firms.states["is_insolvent"] = np.full(n_firms, False)
        test_firms.ts.override_current("equity", np.r_[-1.0, np.full(n_firms - 1, 10.0)])
        test_firms.ts.override_current("deposits", np.r_[-20.0, -30.0, np.zeros(n_firms - 2)])
        illiquid_flag = np.full(n_firms, False)
        illiquid_flag[0] = True

        result = test_firms.handle_insolvency(credit_market=credit_market, illiquid_flag=illiquid_flag)

        assert result.default_flag[0]
        assert not result.default_flag[1]
        assert np.allclose(result.loan_writeoff_by_bank, np.array([100.0, 25.0]))
        assert np.allclose(result.overdraft_writeoff_by_bank, np.array([0.0, 20.0]))
        assert np.allclose(result.credit_loss_by_bank, np.array([100.0, 45.0]))
        assert np.isclose(result.npl_ratio, 125.0 / 175.0)
        assert np.isclose(test_firms.ts.current("deposits")[0], 0.0)
        assert np.isclose(test_firms.ts.current("equity")[0], 0.0)

    def test__handle_insolvency_does_not_double_count_previous_insolvent_state(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        n_banks = 2
        st_loans = np.zeros((3, n_banks, n_firms))
        st_loans[0, 0, 1] = 50.0
        credit_market = CreditMarket.from_data(
            country_name="TST",
            st_loans=st_loans,
            lt_loans=np.zeros((3, n_banks, n_firms)),
            cons_loans=np.zeros((3, n_banks, 1)),
            mort_loans=np.zeros((3, n_banks, 1)),
        )
        test_firms.states["is_insolvent"] = np.r_[False, True, np.full(n_firms - 2, False)]
        test_firms.ts.override_current("equity", np.full(n_firms, 10.0))
        test_firms.ts.override_current("deposits", np.r_[0.0, -30.0, np.zeros(n_firms - 2)])

        result = test_firms.handle_insolvency(
            credit_market=credit_market,
            illiquid_flag=np.full(n_firms, False),
        )

        assert not result.default_flag.any()
        assert np.allclose(result.credit_loss_by_bank, np.zeros(n_banks))
        assert np.isclose(credit_market.states["st_loans"][0, 0, 1], 50.0)
        assert np.isclose(test_firms.ts.current("deposits")[1], -30.0)

    def test__post_credit_mode_does_not_pre_cut_inputs_with_full_budget_credit_gap(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        n_industries = test_firms.n_industries
        unconstrained_intermediate = np.full((n_firms, n_industries), 2.0)
        unconstrained_capital = np.full((n_firms, n_industries), 3.0)

        test_firms.configuration.parameters.firm_activity_finance_revision_mode = "post_credit_cash_budget"
        test_firms.ts.override_current("unconstrained_target_intermediate_inputs", unconstrained_intermediate)
        test_firms.ts.override_current("unconstrained_target_capital_inputs", unconstrained_capital)
        test_firms.ts.override_current("target_short_term_credit", np.full(n_firms, 1_000.0))
        test_firms.ts.override_current("target_long_term_credit", np.full(n_firms, 1_000.0))
        test_firms.ts.override_current("received_short_term_credit", np.zeros(n_firms))
        test_firms.ts.override_current("received_long_term_credit", np.zeros(n_firms))
        test_firms.ts.override_current("deposits", np.full(n_firms, 10_000.0))
        test_firms.ts.override_current("interest_paid", np.zeros(n_firms))
        test_firms.ts.override_current("debt_installments", np.zeros(n_firms))
        test_firms.ts.override_current("planned_technical_investment", np.zeros((n_firms, n_industries)))
        test_firms.ts.override_current("planned_tfp_investment", np.zeros(n_firms))

        test_firms.prepare_buying_goods(
            previous_good_prices=np.ones(n_industries),
            expected_inflation=0.0,
            assume_zero_growth=False,
            wage_obligation_preview=np.zeros(n_firms),
            production_tax_obligation_preview=np.zeros(n_firms),
        )

        assert np.allclose(test_firms.ts.current("target_intermediate_inputs"), unconstrained_intermediate)
        assert np.allclose(test_firms.ts.current("target_capital_inputs"), unconstrained_capital)

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
            (test_firms.ts.initial("target_intermediate_inputs") + test_firms.ts.initial("target_capital_inputs"))
            * (previous_prices * (1 + expected_inflation))[None, :]
        ).sum(axis=1)
        assert np.allclose(
            test_firms.ts.current("planned_intermediate_purchase_expected_costs")
            + test_firms.ts.current("planned_capital_purchase_expected_costs"),
            planned_cost,
        )
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

    def test__set_goods_to_buy_from_current_targets_does_not_append_targets_or_rerun_solver(self, test_firms):
        n_firms = test_firms.ts.current("n_firms")
        n_industries = test_firms.n_industries
        previous_prices = np.full(n_industries, 2.0)
        planned_technical = np.zeros((n_firms, n_industries))
        planned_technical[0, 0] = 6.0
        intermediate_len = len(test_firms.ts.target_intermediate_inputs)
        capital_len = len(test_firms.ts.target_capital_inputs)
        diagnostics_len = len(test_firms.ts.activity_finance_available)

        test_firms.ts.override_current("target_intermediate_inputs", np.ones((n_firms, n_industries)))
        test_firms.ts.override_current("target_capital_inputs", np.full((n_firms, n_industries), 2.0))
        test_firms.ts.override_current("planned_technical_investment", planned_technical)

        test_firms.set_goods_to_buy_from_current_targets(
            previous_good_prices=previous_prices,
            expected_inflation=0.5,
        )

        assert len(test_firms.ts.target_intermediate_inputs) == intermediate_len
        assert len(test_firms.ts.target_capital_inputs) == capital_len
        assert len(test_firms.ts.activity_finance_available) == diagnostics_len
        expected_goods = np.full((n_firms, n_industries), 3.0)
        expected_goods[0, 0] += 2.0
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
