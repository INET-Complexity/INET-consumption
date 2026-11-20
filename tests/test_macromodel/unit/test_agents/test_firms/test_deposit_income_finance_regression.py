"""Regression checks for signed income across firm cash-resource calculations."""

import numpy as np
import pytest


def arrange(firms, monkeypatch, capital=50.0, tfp=0.0, technical=0.0):
    n = firms.ts.current("n_firms")
    m = firms.n_industries

    def full(value):
        return np.full(n, value, dtype=float)

    def matrix(value):
        return np.column_stack([full(value), np.zeros((n, m - 1))])

    values = {
        "deposits": full(100),
        "total_wage": full(80),
        "price": full(1),
        "target_production": full(100),
        "desired_labour_inputs": full(1),
        "taxes_paid_on_production": full(0),
        "debt_installments": full(0),
        "interest_paid": full(0),
        "operating_revolving_interest_paid": full(0),
        "operating_revolving_closing_balance": full(0),
        "operating_revolving_opening_balance": full(0),
        "received_debt_rollover_credit": full(0),
        "received_overdraft_refinance_credit": full(0),
        "received_long_term_credit": full(0),
        "received_short_term_credit": full(0),
        "unconstrained_target_intermediate_inputs_costs": full(0),
        "unconstrained_target_capital_inputs_costs": full(capital),
        "target_intermediate_inputs": matrix(0),
        "target_capital_inputs": matrix(capital),
        "planned_tfp_investment": full(tfp),
        "planned_technical_investment": matrix(technical),
    }
    for key, value in values.items():
        firms.ts.override_current(key, value)
    firms.configuration.parameters.firm_activity_finance_revision_mode = "post_credit_cash_budget"
    monkeypatch.setattr(firms, "capital_formation_tax_on_purchases", lambda value: np.zeros_like(value))
    monkeypatch.setattr(firms, "estimate_corporate_tax_obligation", lambda **kwargs: full(0))
    monkeypatch.setattr(firms, "_activity_finance_tfp_for_feasible_labour", lambda: full(1))
    monkeypatch.setattr(firms, "_activity_finance_feasible_labour", lambda *args: full(1))
    return n, full, matrix


@pytest.mark.parametrize("capital,tfp,technical", [(50, 0, 0), (0, 50, 0), (30, 20, 0), (30, 10, 10)])
@pytest.mark.parametrize("receipt", [0, 10, 50])
def test_full_credit_and_interest_income_fund_the_original_plan(
    test_firms, monkeypatch, capital, tfp, technical, receipt
):
    n, full, _ = arrange(test_firms, monkeypatch, capital, tfp, technical)
    test_firms.compute_target_credit(
        estimated_growth=0,
        estimated_inflation=0,
        interest_obligation_preview=full(-receipt),
        loan_interest_obligation_preview=full(0),
    )
    expected_request = max(0, 30 - receipt)
    np.testing.assert_allclose(test_firms.ts.current("target_long_term_credit"), expected_request)
    test_firms.ts.override_current("received_long_term_credit", full(expected_request))
    test_firms.revise_activity_against_available_finance(
        expected_lcu_prices=np.ones(test_firms.n_industries),
        wage_obligation_preview=full(80),
        production_tax_obligation_preview=full(0),
        interest_obligation_preview=full(-receipt),
        loan_interest_obligation_preview=full(0),
    )
    np.testing.assert_allclose(test_firms.ts.current("target_capital_inputs").sum(axis=1), capital)
    np.testing.assert_allclose(test_firms.ts.current("planned_tfp_investment"), tfp)
    np.testing.assert_allclose(test_firms.ts.current("planned_technical_investment").sum(axis=1), technical)
    np.testing.assert_array_equal(test_firms.ts.current("deposits"), full(100))


def test_partial_credit_preserves_capital_priority_with_receipts(test_firms, monkeypatch):
    _, full, _ = arrange(test_firms, monkeypatch, 30, 20, 0)
    test_firms.ts.override_current("received_long_term_credit", full(5))
    test_firms.revise_activity_against_available_finance(
        expected_lcu_prices=np.ones(test_firms.n_industries),
        wage_obligation_preview=full(80),
        production_tax_obligation_preview=full(0),
        interest_obligation_preview=full(-10),
        loan_interest_obligation_preview=full(0),
    )
    np.testing.assert_allclose(test_firms.ts.current("target_capital_inputs").sum(axis=1), 30)
    np.testing.assert_allclose(test_firms.ts.current("planned_tfp_investment"), 5)


@pytest.mark.parametrize("nonloan,expected", [(-10, 110), (0, 100), (10, 90), (200, 0)])
def test_capacity_consumer_retains_signed_nonloan_interest(test_firms, monkeypatch, nonloan, expected):
    _, full, _ = arrange(test_firms, monkeypatch)
    test_firms.ts.override_current("activity_finance_opening_deposits", full(100))
    test_firms.ts.override_current("target_debt_rollover_credit", full(0))
    test_firms.ts.override_current("target_overdraft_refinance_credit", full(0))
    test_firms.compute_excess_demand_finance_potential_capacities(
        expected_lcu_prices=np.ones(test_firms.n_industries),
        wage_obligation_preview=full(0),
        non_loan_interest_obligation_preview=full(nonloan),
        borrower_st_credit_room=full(0),
        borrower_lt_credit_room=full(0),
        borrower_total_credit_room=full(0),
        q_sold=full(0),
    )
    np.testing.assert_allclose(test_firms._last_excess_demand_finance_potential["finance_cash"], expected)


@pytest.mark.parametrize("nonloan,expected_draw", [(-10, 0), (0, 10), (2, 12)])
def test_operating_draw_accounts_for_signed_receipts(test_firms, monkeypatch, nonloan, expected_draw):
    _, full, _ = arrange(test_firms, monkeypatch)
    test_firms.ts.override_current("deposits", full(5))
    test_firms.ts.override_current("total_wage", full(10))
    result = test_firms.draw_operating_revolving_facility(
        intermediate_purchases=full(5),
        taxes_paid_on_production=full(0),
        non_loan_interest_paid=full(nonloan),
        corporate_tax_obligation_preview=full(0),
    )
    np.testing.assert_allclose(result["current_draw"], expected_draw)
    np.testing.assert_array_equal(test_firms.ts.current("deposits"), full(5))


def test_corrected_planning_and_final_deposit_booking_use_receipt_once(test_firms, monkeypatch):
    _, full, matrix = arrange(test_firms, monkeypatch)
    test_firms.compute_target_credit(
        estimated_growth=0,
        estimated_inflation=0,
        interest_obligation_preview=full(-10),
        loan_interest_obligation_preview=full(0),
    )
    np.testing.assert_allclose(test_firms.ts.current("target_long_term_credit"), 20)
    test_firms.ts.override_current("received_long_term_credit", full(20))
    test_firms.revise_activity_against_available_finance(
        expected_lcu_prices=np.ones(test_firms.n_industries),
        wage_obligation_preview=full(80),
        production_tax_obligation_preview=full(0),
        interest_obligation_preview=full(-10),
        loan_interest_obligation_preview=full(0),
    )
    np.testing.assert_allclose(test_firms.ts.current("target_capital_inputs").sum(axis=1), 50)
    for key in (
        "nominal_amount_sold_in_lcu",
        "corporate_taxes_paid",
        "capital_formation_taxes_paid",
        "operating_revolving_current_draw",
        "received_operating_refinance_credit",
        "operating_revolving_repayment",
        "dividend_fund_settlement_debit",
    ):
        test_firms.ts.override_current(key, full(0))
    test_firms.ts.override_current("received_credit", full(20))
    test_firms.ts.override_current("nominal_amount_spent_in_lcu", matrix(50))
    test_firms.ts.override_current("interest_paid", full(-10))
    monkeypatch.setattr(test_firms, "direct_tfp_investment_cash_expense", lambda: full(0))
    # 100 opening +20 granted +10 receipt -80 operations -50 capital =0.
    np.testing.assert_allclose(test_firms.compute_deposits(), 0)
    np.testing.assert_array_equal(test_firms.ts.current("deposits"), full(100))


def test_unfunded_facility_principal_keeps_existing_residual_cash_priority(test_firms, monkeypatch):
    """Facility principal is not silently promoted ahead of planned investment."""
    _, full, matrix = arrange(test_firms, monkeypatch, capital=20, tfp=0, technical=0)
    test_firms.ts.override_current("operating_revolving_opening_balance", full(50))
    test_firms.ts.override_current("operating_revolving_closing_balance", full(50))
    test_firms.revise_activity_against_available_finance(
        expected_lcu_prices=np.ones(test_firms.n_industries),
        wage_obligation_preview=full(80),
        production_tax_obligation_preview=full(0),
        corporate_tax_obligation_preview=full(0),
        interest_obligation_preview=full(0),
        loan_interest_obligation_preview=full(0),
        debt_installment_preview=full(0),
    )
    np.testing.assert_allclose(test_firms.ts.current("target_capital_inputs").sum(axis=1), 20)
    test_firms.ts.override_current("operating_revolving_current_draw", full(0))
    # Realising the full 80 operating +20 investment plan consumes opening cash100.
    result = test_firms.settle_operating_revolving_facility(full(0))
    np.testing.assert_allclose(result["repayment"], 0)
    np.testing.assert_allclose(result["closing_balance"], 50)
