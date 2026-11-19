"""Contractual payment cash flow, independent of rate/stock approximations."""

from copy import deepcopy

import numpy as np
import pytest

from macromodel.agents.households.cashflow_rates import DebtServicePayments
from macromodel.agents.households.func.consumption import CreditAugmentedConsumption
from macromodel.markets.credit_market.credit_market import CreditMarket


def market(consumer=(1000, 0.01, 100), mortgage=(2000, 0.02, 200)):
    credit = object.__new__(CreditMarket)
    credit._serviceable_loans_this_period = {
        "cons_loans": np.asarray(consumer, dtype=float).reshape(3, 1, 1),
        "mort_loans": np.asarray(mortgage, dtype=float).reshape(3, 1, 1),
    }
    credit._consumer_principal_arrears_by_cell = np.zeros((1, 1))
    credit._mortgage_principal_arrears_by_cell = np.zeros((1, 1))
    credit._consumer_interest_arrears_by_cell = np.zeros((1, 1))
    credit._mortgage_interest_arrears_by_cell = np.zeros((1, 1))
    return credit


def service(credit):
    return sum(credit.preview_opening_contractual_household_service().values())


def test_contractual_accessor_separates_both_products_interest_and_principal_without_writes():
    credit = market()
    before = deepcopy(credit.__dict__)
    components = credit.preview_opening_contractual_household_service()
    np.testing.assert_array_equal(components["consumer_interest_nominal"], [10])
    np.testing.assert_array_equal(components["consumer_principal_nominal"], [90])
    np.testing.assert_array_equal(components["mortgage_interest_nominal"], [40])
    np.testing.assert_array_equal(components["mortgage_principal_nominal"], [160])
    for key, value in before.items():
        if isinstance(value, dict):
            for name in value:
                np.testing.assert_array_equal(credit.__dict__[key][name], value[name])
        else:
            np.testing.assert_array_equal(credit.__dict__[key], value)


def test_fixed_annuity_amortisation_and_rate_change_without_payment_change_have_zero_shock():
    previous = service(market())
    amortised = service(market(consumer=(910, 0.01, 100), mortgage=(1840, 0.02, 200)))
    new_interest_same_annuity = service(market(consumer=(1000, 0.02, 100), mortgage=(2000, 0.03, 200)))
    np.testing.assert_allclose(DebtServicePayments(amortised, previous).change_ratio([1000]), [0])
    np.testing.assert_allclose(DebtServicePayments(new_interest_same_annuity, previous).change_ratio([1000]), [0])


def test_variable_payment_reset_and_refinancing_follow_actual_schedule_amounts():
    previous = service(market())
    reset = service(market(consumer=(1000, 0.02, 110), mortgage=(2000, 0.03, 220)))
    refinanced = service(market(consumer=(1000, 0.005, 70), mortgage=(2000, 0.02, 150)))
    np.testing.assert_allclose(-0.75 * DebtServicePayments(reset, previous).change_ratio([1000]), [-0.0225])
    np.testing.assert_allclose(-0.75 * DebtServicePayments(refinanced, previous).change_ratio([1000]), [0.06])


def test_final_payment_is_capped_then_loan_maturity_releases_cash_flow():
    last = service(market(consumer=(40, 0.01, 100), mortgage=(0, 0, 0)))
    np.testing.assert_allclose(last, [40.4])
    next_payment = service(market(consumer=(0, 0.01, 100), mortgage=(0, 0.02, 200)))
    np.testing.assert_array_equal(next_payment, [0])
    np.testing.assert_allclose(-0.75 * DebtServicePayments(next_payment, last).change_ratio([1000]), [0.0303])


def test_old_arrears_and_voluntary_early_repayment_are_not_new_contractual_instalments():
    credit = market(consumer=(1000, 0.01, 100), mortgage=(2000, 0.02, 200))
    credit._consumer_principal_arrears_by_cell[:] = 50
    credit._consumer_interest_arrears_by_cell[:] = 10
    credit._mortgage_principal_arrears_by_cell[:] = 100
    credit._mortgage_interest_arrears_by_cell[:] = 20
    credit._consumer_payment_settlement = object()  # settled/early-paid cash is not read by the accessor
    np.testing.assert_allclose(service(credit), [300])
    consumer_due, mortgage_due = credit.preview_opening_household_service()
    np.testing.assert_allclose(consumer_due + mortgage_due, [480])


def test_nominal_flow_units_do_not_annualise_compound_or_deflate_the_payment_difference():
    payment = DebtServicePayments([150, 0, 100], [100, 0, 150])
    income = np.array([1000, 50, 1000])
    np.testing.assert_allclose(-0.75 * payment.change_ratio(income), [-0.0375, 0, 0.0375])
    scaled = DebtServicePayments(payment.current_nominal * 1e6, payment.previous_nominal * 1e6)
    np.testing.assert_allclose(scaled.change_ratio(income * 1e6), payment.change_ratio(income))
    # A fixed nominal payment has no payment shock, regardless of CPI or income growth.
    np.testing.assert_array_equal(DebtServicePayments([100], [100]).change_ratio([2000]), [0])
    np.testing.assert_array_equal(DebtServicePayments([100], [0]).change_ratio([1000]), [0.1])


@pytest.mark.parametrize(
    "current,previous,income",
    [
        ([-1], [0], [1]),
        ([0], [-1], [1]),
        ([np.nan], [0], [1]),
        ([0], [np.inf], [1]),
        ([0], [0], [0]),
        ([0], [0], [-1]),
        ([0], [0], [np.nan]),
        ([1], [1, 2], [1]),
    ],
)
def test_invalid_payment_records_fail(current, previous, income):
    with pytest.raises(ValueError):
        DebtServicePayments(current, previous).change_ratio(income)


def test_payment_inputs_are_copied_and_immutable():
    source = np.array([100.0])
    payment = DebtServicePayments(source, [50])
    source[0] = 900
    assert payment.current_nominal[0] == 100
    with pytest.raises(ValueError):
        payment.current_nominal[0] = 1


@pytest.mark.parametrize("coefficient", [np.nan, np.inf, 0.75])
def test_payment_coefficient_rejects_invalid_sign_and_values(coefficient):
    with pytest.raises(ValueError, match="nonpositive"):
        CreditAugmentedConsumption(debt_service_cashflow_propensity=coefficient)
