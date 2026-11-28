"""Switch `charge_direct_tfp_investment`: True charges direct TFP investment to firm cash, False does not."""

import numpy as np
import pytest

from macromodel.agents.firms.func.productivity_growth import SimpleTFPGrowth


class IndependentTFPPlanner:
    executes_direct_tfp_independently = True


def set_charge(firms, charge: bool) -> None:
    firms.configuration.parameters.charge_direct_tfp_investment = charge


def test_default_charges_direct_tfp_investment(test_firms):
    assert test_firms.configuration.parameters.charge_direct_tfp_investment is True
    assert test_firms.charges_direct_tfp_investment()


@pytest.mark.parametrize("charge,expected_delta", [(True, -18.0), (False, -14.0)])
def test_profits_charge_direct_tfp_only_when_switch_on(test_firms, charge, expected_delta):
    set_charge(test_firms, charge)
    test_firms.configuration.parameters.capital_compensation_accounting_mode = "surplus_pool"
    test_firms.ts.used_intermediate_inputs_costs.append(np.full(18, 10.0))
    test_firms.ts.used_capital_inputs_costs.append(np.full(18, 10.0))
    test_firms.ts.taxes_paid_on_production.append(np.full(18, 1.0))
    test_firms.ts.interest_paid.append(np.full(18, 2.0))
    test_firms.ts.total_wage.append(np.full(18, 1.0))
    test_firms.ts.direct_tfp_investment_cash_expense.append(np.full(18, 4.0))
    revenue = test_firms.ts.current("production") * test_firms.ts.current("price")

    assert np.allclose(test_firms.compute_profits() - revenue, expected_delta)
    assert np.allclose(
        test_firms.compute_profits_given_interest_paid(interest_paid=np.full(18, 2.0)) - revenue,
        expected_delta,
    )


@pytest.mark.parametrize("charge,expected", [(True, 114.0), (False, 121.0)])
def test_deposits_charge_direct_tfp_only_when_switch_on(test_firms, charge, expected):
    set_charge(test_firms, charge)
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

    assert np.allclose(test_firms.compute_deposits(), expected)
    expected_expense = 7.0 if charge else 0.0
    assert np.allclose(test_firms.direct_tfp_investment_cash_expense(), expected_expense)


@pytest.mark.parametrize(
    "charge,tfp_costs,investment_budget,long_term_credit",
    [(True, 3.0, 13.0, 13.0), (False, 0.0, 10.0, 10.0)],
)
def test_credit_budget_excludes_uncharged_tfp(test_firms, charge, tfp_costs, investment_budget, long_term_credit):
    set_charge(test_firms, charge)
    n_firms = test_firms.ts.current("n_firms")
    n_industries = test_firms.n_industries
    zeros = np.zeros(n_firms - 1)
    test_firms.ts.override_current("deposits", np.r_[10.0, zeros])
    test_firms.ts.override_current("price", np.r_[2.0, np.ones(n_firms - 1)])
    test_firms.ts.override_current("target_production", np.r_[10.0, zeros])
    test_firms.ts.override_current("corporate_taxes_paid", np.zeros(n_firms))
    test_firms.ts.override_current("interest_paid", np.r_[4.0, zeros])
    test_firms.ts.override_current("debt_installments", np.r_[1.0, zeros])
    test_firms.ts.override_current("unconstrained_target_intermediate_inputs_costs", np.r_[12.0, zeros])
    test_firms.ts.override_current("unconstrained_target_capital_inputs_costs", np.r_[8.0, zeros])
    planned_technical = np.zeros((n_firms, n_industries))
    planned_technical[0, :2] = [1.5, 0.5]
    test_firms.ts.override_current("planned_technical_investment", planned_technical)
    test_firms.ts.override_current("planned_tfp_investment", np.r_[3.0, zeros])

    test_firms.compute_target_credit(
        estimated_growth=0.0,
        estimated_inflation=0.0,
        wage_obligation_preview=np.r_[15.0, zeros],
        production_tax_obligation_preview=np.r_[5.0, zeros],
    )

    assert np.isclose(test_firms.ts.current("credit_budget_tfp_costs")[0], tfp_costs)
    assert np.isclose(test_firms.ts.current("credit_budget_investment_budget")[0], investment_budget)
    assert np.isclose(test_firms.ts.current("target_long_term_credit")[0], long_term_credit)
    assert np.isclose(test_firms.ts.current("target_short_term_credit")[0], 1.0)
    # The planned amount itself is never altered by the credit budget.
    assert np.isclose(test_firms.ts.current("planned_tfp_investment")[0], 3.0)


def _set_post_credit_revision_inputs(test_firms) -> None:
    n_firms = test_firms.ts.current("n_firms")
    n_industries = test_firms.n_industries
    firm_industry = test_firms.states["Industry"][0]
    intermediate_coefficients = np.zeros((n_industries, n_industries))
    intermediate_coefficients[0, firm_industry] = 1.0
    test_firms.base_intermediate_inputs_productivity_matrix = intermediate_coefficients
    test_firms.base_capital_input_use_matrix = np.zeros((n_industries, n_industries))
    test_firms.ts.override_current("target_production", np.r_[20.0, np.zeros(n_firms - 1)])
    test_firms.ts.override_current("tfp_multiplier", np.ones(n_firms))
    test_firms.ts.override_current("limiting_intermediate_inputs", np.full(n_firms, 1_000.0))
    test_firms.ts.override_current("limiting_capital_inputs", np.full(n_firms, 1_000.0))
    test_firms.ts.override_current("intermediate_inputs_stock", np.zeros((n_firms, n_industries)))
    test_firms.ts.override_current("capital_inputs_stock", np.zeros((n_firms, n_industries)))

    target_intermediate = np.zeros((n_firms, n_industries))
    planned_technical = np.zeros((n_firms, n_industries))
    target_intermediate[0, 0] = 20.0
    planned_technical[0, 0] = 5.0
    test_firms.configuration.parameters.firm_activity_finance_revision_mode = "post_credit_cash_budget"
    test_firms.ts.override_current("deposits", np.zeros(n_firms))
    test_firms.ts.override_current("received_short_term_credit", np.r_[15.0, np.zeros(n_firms - 1)])
    test_firms.ts.override_current("received_long_term_credit", np.zeros(n_firms))
    test_firms.ts.override_current("interest_paid", np.zeros(n_firms))
    test_firms.ts.override_current("debt_installments", np.zeros(n_firms))
    test_firms.ts.override_current("target_intermediate_inputs", target_intermediate)
    test_firms.ts.override_current("target_capital_inputs", np.zeros((n_firms, n_industries)))
    test_firms.ts.override_current("planned_technical_investment", planned_technical)
    test_firms.ts.override_current("planned_tfp_investment", np.r_[5.0, np.zeros(n_firms - 1)])
    test_firms.ts.override_current("unconstrained_target_intermediate_inputs", np.full((n_firms, n_industries), 99.0))


@pytest.mark.parametrize("charge,expected_tfp,expected_scale", [(True, 0.0, 0.0), (False, 5.0, 1.0)])
def test_finance_revision_scales_tfp_only_when_charged(test_firms, charge, expected_tfp, expected_scale):
    set_charge(test_firms, charge)
    _set_post_credit_revision_inputs(test_firms)
    n_firms = test_firms.ts.current("n_firms")

    test_firms.revise_activity_against_available_finance(
        expected_lcu_prices=np.ones(test_firms.n_industries),
        wage_obligation_preview=np.zeros(n_firms),
        production_tax_obligation_preview=np.zeros(n_firms),
    )

    # Technical investment still needs finance and is cut to zero in both cases.
    assert np.isclose(test_firms.ts.current("planned_technical_investment")[0, 0], 0.0)
    assert np.isclose(test_firms.ts.current("planned_tfp_investment")[0], expected_tfp)
    assert np.isclose(test_firms.ts.current("tfp_investment_finance_scale")[0], expected_scale, atol=1e-12)
    assert np.isclose(test_firms.ts.current("planned_productivity_investment")[0], expected_tfp)
    assert test_firms.ts.current("activity_finance_feasibility_residual")[0] >= -1e-6
    technical_costs = test_firms.ts.current("planned_technical_investment_expected_costs")[0]
    assert np.isclose(test_firms.ts.current("activity_finance_gap_before_revision")[0], technical_costs + 5.0 * charge)


@pytest.mark.parametrize("charge", [True, False])
def test_execution_and_tfp_growth_do_not_depend_on_charge(test_firms, charge):
    set_charge(test_firms, charge)
    n_firms = test_firms.ts.current("n_firms")
    n_industries = test_firms.n_industries
    planned_tfp = np.full(n_firms, 8.0)  # the planner's cash-limited amount
    test_firms.functions["productivity_investment_planner"] = IndependentTFPPlanner()
    test_firms.functions["productivity_growth"] = SimpleTFPGrowth(investment_effectiveness=0.1)
    test_firms.configuration.parameters.tfp_base_growth_rate = 0.0
    test_firms.configuration.parameters.tfp_investment_elasticity = 0.3
    test_firms.current_good_prices = np.ones(n_industries)
    test_firms.base_capital_input_use_matrix = np.zeros((n_industries, n_industries))
    test_firms.ts.production.append(np.ones(n_firms))
    test_firms.ts.price.append(np.full(n_firms, 2.0))
    test_firms.ts.total_capital_inputs_bought_costs.append(np.zeros(n_firms))
    test_firms.ts.planned_productivity_investment.append(planned_tfp.copy())
    test_firms.ts.planned_tfp_investment.append(planned_tfp.copy())
    test_firms.ts.planned_technical_investment.append(np.zeros((n_firms, n_industries)))

    test_firms.execute_productivity_investment()

    assert np.allclose(test_firms.ts.current("executed_tfp_investment"), planned_tfp)
    expected_expense = planned_tfp if charge else np.zeros(n_firms)
    assert np.allclose(test_firms.ts.current("direct_tfp_investment_cash_expense"), expected_expense)
    assert np.allclose(test_firms.direct_tfp_investment_cash_expense(), expected_expense)
    tfp_growth = test_firms.compute_tfp_growth()
    assert np.all(tfp_growth > 0.0)
    assert np.allclose(tfp_growth, 0.1 * (8.0 / 2.0) ** 0.3)


def test_flow_residual_control_passes_with_uncharged_tfp(test_firms):
    set_charge(test_firms, False)
    n_firms = test_firms.ts.current("n_firms")
    n_industries = test_firms.n_industries
    test_firms.functions["productivity_investment_planner"] = IndependentTFPPlanner()
    test_firms.current_good_prices = np.ones(n_industries)
    test_firms.base_capital_input_use_matrix = np.zeros((n_industries, n_industries))
    test_firms.ts.production.append(np.ones(n_firms))
    test_firms.ts.total_capital_inputs_bought_costs.append(np.zeros(n_firms))
    test_firms.ts.planned_productivity_investment.append(np.full(n_firms, 8.0))
    test_firms.ts.planned_tfp_investment.append(np.full(n_firms, 8.0))
    test_firms.ts.planned_technical_investment.append(np.zeros((n_firms, n_industries)))
    test_firms.execute_productivity_investment()
    assert np.allclose(test_firms.ts.current("executed_tfp_investment"), 8.0)

    opening_deposits = np.full(n_firms, 100.0)
    nominal_amount_sold_in_lcu = np.full(n_firms, 20.0)
    received_credit = np.full(n_firms, 10.0)
    total_wage = np.full(n_firms, 5.0)
    taxes_paid_on_production = np.full(n_firms, 2.0)
    corporate_taxes_paid = np.full(n_firms, 3.0)
    interest_paid = np.full(n_firms, 4.0)
    debt_installments = np.full(n_firms, 6.0)
    # Executed TFP investment (8.0) is not debited from closing deposits.
    closing_deposits = (
        opening_deposits
        + nominal_amount_sold_in_lcu
        + received_credit
        - total_wage
        - taxes_paid_on_production
        - corporate_taxes_paid
        - interest_paid
        - debt_installments
    )
    test_firms.ts.override_current("activity_finance_opening_deposits", opening_deposits)
    test_firms.ts.override_current("deposits", closing_deposits)
    test_firms.ts.override_current("nominal_amount_sold_in_lcu", nominal_amount_sold_in_lcu)
    test_firms.ts.override_current("received_credit", received_credit)
    test_firms.ts.override_current("total_wage", total_wage)
    test_firms.ts.override_current("nominal_amount_spent_in_lcu", np.zeros((n_firms, n_industries)))
    test_firms.ts.override_current("taxes_paid_on_production", taxes_paid_on_production)
    test_firms.ts.override_current("corporate_taxes_paid", corporate_taxes_paid)
    test_firms.ts.override_current("interest_paid", interest_paid)
    test_firms.ts.override_current("debt_installments", debt_installments)
    test_firms.ts.override_current("inventory", np.ones(n_firms))
    test_firms.ts.override_current("price", np.full(n_firms, 2.0))
    test_firms.ts.override_current("intermediate_inputs_stock", np.zeros((n_firms, n_industries)))
    test_firms.ts.override_current("capital_inputs_stock", np.zeros((n_firms, n_industries)))
    test_firms.ts.override_current("debt", np.full(n_firms, 5.0))
    test_firms.ts.override_current("operating_revolving_closing_balance", np.full(n_firms, 8.0))
    test_firms.ts.override_current("equity", np.full(n_firms, 99.0))

    result = test_firms.check_firm_accounting_controls(current_good_prices=np.ones(n_industries), enforce=True)

    assert np.allclose(result["transaction_flow_residual"], 0.0)
    assert np.all(result["control_passed"])
