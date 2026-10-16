from types import SimpleNamespace

import numpy as np
import pytest

from macromodel.markets.credit_market.credit_market import CreditMarket
from macromodel.markets.credit_market.func.clearing import _annuity_payment_factor


def market():
    cons = np.zeros((3, 2, 2))
    cons[:, 0, 0] = [60, 0.01, 6]
    cons[:, 1, 0] = [40, 0.02, 5]
    mortgage = np.zeros_like(cons)
    mortgage[:, 0, 0] = [100, 0.01, 10]
    return CreditMarket.from_data("TST", np.zeros((3, 2, 1)), np.zeros((3, 2, 1)), cons, mortgage)


def household(income=(100, 100), ficp=(False, False)):
    return SimpleNamespace(
        consumer_underwriting_income=np.array(income, dtype=float),
        consumer_underwriting_other_service=np.array([2.0, 0.0]),
        consumer_underwriting_dsti_limit=0.35,
        uses_feasibility_resolver=True,
        current_ficp_active=lambda: np.array(ficp),
    )


def test_actual_rate_and_total_balance_constrain_incremental_cash_and_mortgages():
    m = market()
    new = np.zeros((3, 2, 2))
    new[:, 0, 0] = [1000, 0.05, 100]
    mortgage = np.zeros_like(new)
    mortgage[:, 0, 0] = [50, 0.02, 20]
    old = m.states["cons_loans"].copy()
    cons, mort = m.enforce_household_origination_dsti(household(), 12, new, mortgage)
    principal = cons[0, :, 0].sum()
    # Current opening consumer interest is reserved if payments are suspended.
    assert (100 + 1.4 + principal) * _annuity_payment_factor(np.array([0.05]), 12)[0] + 10 + 2 == pytest.approx(35)
    assert mort[0].sum() == 0  # all-or-nothing mortgage exceeds combined capacity
    np.testing.assert_array_equal(m.states["cons_loans"], old)


@pytest.mark.parametrize("income,ficp", [((0, -1), (False, False)), ((100, 100), (True, True))])
def test_nonpositive_income_and_ficp_cannot_authorise_new_cash(income, ficp):
    m = market()
    new = np.zeros((3, 2, 2))
    new[0] = 100
    cons, _ = m.enforce_household_origination_dsti(household(income, ficp), 12, new, np.zeros_like(new))
    assert cons[0].sum() == 0


def test_refinancing_preserves_each_lenders_claim_and_issues_no_second_cash_receipt():
    m = market()
    m._consumer_interest_arrears_by_cell[:, 0] = [3, 2]
    m._consumer_principal_arrears_by_cell[:, 0] = [5, 2]
    new = np.zeros((3, 2, 2))
    new[:, 1, 0] = [20, 0.03, 2]
    m._add_new_loans("cons_loans", new)
    debt_before = m.compute_outstanding_consumption_loans_by_household().copy()
    bank_claims = m.compute_outstanding_loans_by_bank().copy()
    unchanged_nonborrower = m.states["cons_loans"][:, :, 1].copy()
    m._remodulate_settled_consumer_loan_schedule(settled_loans=new, consumer_loan_maturity=12)
    np.testing.assert_allclose(m.compute_outstanding_consumption_loans_by_household(), debt_before)
    np.testing.assert_allclose(m.compute_outstanding_loans_by_bank(), bank_claims)
    np.testing.assert_array_equal(m.states["cons_loans"][:, :, 1], unchanged_nonborrower)
    np.testing.assert_allclose(m.states["cons_loans"][0, :, 0], [63, 62])
    assert m._consumer_interest_arrears_by_cell.sum() == 0
    assert m._consumer_principal_arrears_by_cell.sum() == 0
    assert m.states["cons_loans"][2, :, 0].sum() == pytest.approx(
        125 * _annuity_payment_factor(np.array([0.03]), 12)[0]
    )
