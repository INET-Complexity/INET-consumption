"""Acquisition-tax payments must match receipts and constrain available cash."""

import numpy as np
import pytest

from macromodel.agents.individuals.individual_properties import ActivityStatus
from macromodel.markets.credit_market.func.clearing import _compute_firm_cfads


@pytest.mark.parametrize("rate", [0.0, 0.07, 0.25])
def test_acquisition_tax_matches_government_and_firm_cash(test_firms, test_central_government, rate):
    firms, government = test_firms, test_central_government
    n = firms.ts.current("n_firms")
    purchases = np.arange(n, dtype=float) * 2.0
    firms.states["Capital Formation Tax"] = rate
    for key in (
        "nominal_amount_sold_in_lcu",
        "total_wage",
        "direct_tfp_investment_cash_expense",
        "taxes_paid_on_production",
        "corporate_taxes_paid",
        "interest_paid",
        "received_credit",
        "operating_revolving_current_draw",
        "received_operating_refinance_credit",
        "operating_revolving_repayment",
        "debt_installments",
        "dividend_fund_settlement_debit",
    ):
        firms.ts.override_current(key, np.zeros(n))
    firms.ts.override_current("deposits", np.full(n, 100.0))
    spending = np.zeros((n, firms.n_industries))
    spending[:, 0] = purchases
    firms.ts.override_current("nominal_amount_spent_in_lcu", spending)
    firms.ts.override_current("capital_formation_taxes_paid", firms.capital_formation_tax_on_purchases(purchases))
    closing = firms.compute_deposits()

    government.states["Household Capital Formation Tax"] = 0.0
    government.states["Firm Capital Formation Tax"] = rate
    government.compute_taxes(
        current_ind_employee_income=np.zeros(1),
        current_total_rent_paid=0.0,
        current_income_financial_assets=np.zeros(1),
        current_ind_activity=np.array([ActivityStatus.EMPLOYED]),
        current_ind_realised_cons=np.zeros(1),
        current_bank_profits=np.zeros(1),
        current_firm_production=np.zeros(n),
        current_firm_price=np.ones(n),
        current_firm_profits=np.zeros(n),
        current_firm_industries=np.zeros(n, dtype=int),
        current_household_new_real_wealth=np.zeros(1),
        taxes_less_subsidies_rates=np.zeros(1),
        current_total_exports=0.0,
        current_firm_capital_formation=purchases.sum(),
    )
    np.testing.assert_allclose(closing, 100.0 - purchases * (1.0 + rate))
    assert (100.0 - purchases - closing).sum() == pytest.approx(government.ts.current("taxes_cf")[0])

    # The independent transaction-flow control must include the same debit.
    firms.ts.override_current("activity_finance_opening_deposits", np.full(n, 100.0))
    firms.ts.override_current("deposits", closing)
    firms.ts.override_current("equity", firms.compute_equity(np.ones(firms.n_industries)))
    controls = firms.check_firm_accounting_controls(np.ones(firms.n_industries))
    np.testing.assert_allclose(controls["transaction_flow_residual"], 0.0, atol=1e-10)


@pytest.mark.parametrize("technical, tfp, credit", [(0.0, 0.0, 50.0), (20.0, 5.0, 80.0)])
def test_investment_credit_request_includes_acquisition_tax(test_firms, technical, tfp, credit):
    firms = test_firms
    n = firms.ts.current("n_firms")
    firms.states["Capital Formation Tax"] = 0.25
    for key in (
        "deposits",
        "interest_paid",
        "debt_installments",
        "corporate_taxes_paid",
        "operating_revolving_closing_balance",
        "unconstrained_target_intermediate_inputs_costs",
        "planned_tfp_investment",
    ):
        firms.ts.override_current(key, np.zeros(n))
    firms.ts.override_current("unconstrained_target_capital_inputs_costs", np.full(n, 40.0))
    technical_plan = np.zeros((n, firms.n_industries))
    technical_plan[:, 0] = technical
    firms.ts.override_current("planned_technical_investment", technical_plan)
    firms.ts.override_current("planned_tfp_investment", np.full(n, tfp))
    firms.compute_target_credit(
        0.0, 0.0, wage_obligation_preview=np.zeros(n), production_tax_obligation_preview=np.zeros(n)
    )
    np.testing.assert_allclose(firms.ts.current("target_long_term_credit"), credit)


@pytest.mark.parametrize("rate, expected", [(0.0, 40.0), (0.25, 32.0)])
def test_finance_revision_reserves_tax_without_inflating_goods_orders(test_firms, rate, expected):
    firms = test_firms
    n = firms.ts.current("n_firms")
    firms.states["Capital Formation Tax"] = rate
    firms.configuration.parameters.firm_activity_finance_revision_mode = "post_credit_cash_budget"
    capital = np.zeros((n, firms.n_industries))
    capital[:, 0] = 40.0
    firms.ts.override_current("deposits", np.full(n, 40.0))
    for key in (
        "interest_paid",
        "debt_installments",
        "received_debt_rollover_credit",
        "received_long_term_credit",
        "planned_tfp_investment",
    ):
        firms.ts.override_current(key, np.zeros(n))
    for key in ("target_intermediate_inputs", "planned_technical_investment"):
        firms.ts.override_current(key, np.zeros_like(capital))
    firms.ts.override_current("target_capital_inputs", capital)
    firms.revise_activity_against_available_finance(np.ones(firms.n_industries), np.zeros(n), np.zeros(n))
    np.testing.assert_allclose(firms.ts.current("target_capital_inputs")[:, 0], expected)
    np.testing.assert_allclose(firms.ts.current("activity_finance_feasibility_residual"), 0.0)


def test_recorded_acquisition_tax_reduces_cfads(test_firms):
    firms = test_firms
    n = firms.ts.current("n_firms")
    for key in ("total_wage", "taxes_paid_on_production", "corporate_taxes_paid"):
        firms.ts.override_current(key, np.zeros(n))
    firms.ts.override_current("nominal_amount_sold_in_lcu", np.full(n, 100.0))
    firms.ts.override_current("nominal_amount_spent_in_lcu", np.zeros((n, firms.n_industries)))
    firms.ts.override_current("capital_formation_taxes_paid", np.full(n, 10.0))
    np.testing.assert_allclose(_compute_firm_cfads(firms, window=1, haircut=0.8), 72.0)


@pytest.mark.parametrize("rate, payable", [(0.0, 20.0), (0.25, 10.0)])
def test_acquisition_tax_is_reserved_before_debt_service(test_firms, rate, payable):
    firms = test_firms
    n = firms.ts.current("n_firms")
    firms.states["Capital Formation Tax"] = rate
    firms.begin_firm_debt_settlement(
        opening_interest_arrears=np.zeros(n),
        opening_principal_arrears=np.zeros(n),
        contractual_interest_due=np.zeros(n),
        contractual_principal_due=np.full(n, 40.0),
        scheduled_interest_due=np.zeros(n),
        scheduled_principal_due=np.full(n, 40.0),
    )
    firms.ts.override_current("deposits", np.full(n, 60.0))
    for key in (
        "nominal_amount_sold_in_lcu",
        "total_wage",
        "direct_tfp_investment_cash_expense",
        "received_ordinary_short_term_credit",
        "received_debt_rollover_credit",
        "received_overdraft_refinance_credit",
        "received_long_term_credit",
        "operating_revolving_current_draw",
    ):
        firms.ts.override_current(key, np.zeros(n))
    spending = np.zeros((n, firms.n_industries))
    spending[:, 0] = 40.0
    firms.ts.override_current("nominal_amount_spent_in_lcu", spending)
    firms.ts.override_current(
        "capital_formation_taxes_paid", firms.capital_formation_tax_on_purchases(np.full(n, 40.0))
    )
    settlement = firms.plan_firm_debt_settlement(np.zeros(n), np.zeros(n), np.zeros(n))
    np.testing.assert_allclose(settlement["payable_principal"], payable)
    np.testing.assert_allclose(settlement["closing_principal_arrears"], 40.0 - payable)


def test_finance_revision_reserves_tax_on_technical_goods_but_not_direct_tfp(test_firms):
    firms = test_firms
    n = firms.ts.current("n_firms")
    firms.states["Capital Formation Tax"] = 0.25
    firms.configuration.parameters.firm_activity_finance_revision_mode = "post_credit_cash_budget"
    capital = np.zeros((n, firms.n_industries))
    capital[:, 0] = 20.0
    firms.ts.override_current("deposits", np.full(n, 40.0))
    for key in ("interest_paid", "debt_installments", "received_debt_rollover_credit", "received_long_term_credit"):
        firms.ts.override_current(key, np.zeros(n))
    firms.ts.override_current("target_intermediate_inputs", np.zeros_like(capital))
    firms.ts.override_current("target_capital_inputs", capital)
    firms.ts.override_current("planned_technical_investment", capital.copy())
    firms.ts.override_current("planned_tfp_investment", np.full(n, 5.0))
    firms.revise_activity_against_available_finance(np.ones(firms.n_industries), np.zeros(n), np.zeros(n))
    # Capital first costs 25; remaining 15 funds half the 25+5 productivity budget.
    np.testing.assert_allclose(firms.ts.current("target_capital_inputs")[:, 0], 20.0)
    np.testing.assert_allclose(firms.ts.current("planned_technical_investment")[:, 0], 10.0)
    np.testing.assert_allclose(firms.ts.current("planned_tfp_investment"), 2.5)
    np.testing.assert_allclose(firms.ts.current("activity_finance_feasibility_residual"), 0.0)
