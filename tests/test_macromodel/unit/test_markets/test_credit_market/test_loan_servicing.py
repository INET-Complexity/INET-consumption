import numpy as np

from macromodel.markets.credit_market.credit_market import CreditMarket
from macromodel.markets.credit_market.func.clearing import _annuity_payment_factor


def _loan_array(n_banks: int = 2, n_borrowers: int = 2) -> np.ndarray:
    return np.zeros((3, n_banks, n_borrowers))


def test_annuity_payment_factor_positive_and_zero_rate():
    assert np.isclose(_annuity_payment_factor(0.05, 4), 0.2820118326037872)
    assert np.isclose(_annuity_payment_factor(0.0, 4), 0.25)


def test_firm_interest_is_recomputed_on_outstanding_principal():
    st_loans = _loan_array()
    st_loans[0, 0, 0] = 100.0
    st_loans[1, 0, 0] = 0.05
    st_loans[2, 0, 0] = 100.0 * _annuity_payment_factor(0.05, 4)
    market = CreditMarket.from_data(
        country_name="TST",
        st_loans=st_loans,
        lt_loans=_loan_array(),
        cons_loans=_loan_array(),
        mort_loans=_loan_array(),
    )

    first_principal = market.pay_firm_installments()
    first_interest = market.compute_interest_paid_by_firm()
    second_principal = market.pay_firm_installments()
    second_interest = market.compute_interest_paid_by_firm()

    assert np.isclose(first_interest[0], 5.0)
    assert np.isclose(first_principal[0], st_loans[2, 0, 0] - 5.0)
    assert second_interest[0] < first_interest[0]
    assert second_principal[0] > first_principal[0]


def test_newly_originated_firm_loan_service_starts_next_quarter():
    market = CreditMarket.from_data(
        country_name="TST",
        st_loans=_loan_array(),
        lt_loans=_loan_array(),
        cons_loans=_loan_array(),
        mort_loans=_loan_array(),
    )
    market._serviceable_loans_this_period = {
        key: market.states[key].copy() for key in market._serviceable_loans_this_period
    }

    new_st_loan = _loan_array()
    new_st_loan[0, 0, 0] = 100.0
    new_st_loan[1, 0, 0] = 0.05
    new_st_loan[2, 0, 0] = 100.0 * _annuity_payment_factor(0.05, 4)
    market._add_new_loans("st_loans", new_st_loan)

    same_quarter_principal = market.pay_firm_installments()
    same_quarter_interest = market.compute_interest_paid_by_firm()

    assert same_quarter_principal[0] == 0.0
    assert same_quarter_interest[0] == 0.0
    assert market.states["st_loans"][0, 0, 0] == 100.0

    next_quarter_principal = market.pay_firm_installments()
    next_quarter_interest = market.compute_interest_paid_by_firm()

    assert np.isclose(next_quarter_interest[0], 5.0)
    assert next_quarter_principal[0] > 0.0


def test_bank_interest_income_matches_borrower_loan_interest():
    st_loans = _loan_array()
    st_loans[0, 0, 0] = 100.0
    st_loans[1, 0, 0] = 0.05
    st_loans[2, 0, 0] = 100.0 * _annuity_payment_factor(0.05, 4)
    cons_loans = _loan_array()
    cons_loans[0, 0, 1] = 50.0
    cons_loans[1, 0, 1] = 0.04
    cons_loans[2, 0, 1] = 50.0 * _annuity_payment_factor(0.04, 4)
    market = CreditMarket.from_data(
        country_name="TST",
        st_loans=st_loans,
        lt_loans=_loan_array(),
        cons_loans=cons_loans,
        mort_loans=_loan_array(),
    )

    market.pay_firm_installments()
    firm_interest = market.compute_interest_paid_by_firm()
    market.pay_household_installments()
    household_interest = market.compute_interest_paid_by_household()
    bank_interest = market.compute_interest_received_by_bank()

    assert np.isclose(bank_interest.sum(), firm_interest.sum() + household_interest.sum())


def test_partial_firm_settlement_is_prorated_and_preserves_unpaid_principal():
    st_loans = _loan_array()
    st_loans[0, 0, 0] = 100.0
    st_loans[1, 0, 0] = 0.10
    st_loans[2, 0, 0] = 30.0
    lt_loans = _loan_array()
    lt_loans[0, 1, 0] = 50.0
    lt_loans[1, 1, 0] = 0.10
    lt_loans[2, 1, 0] = 20.0
    market = CreditMarket.from_data(
        country_name="TST",
        st_loans=st_loans,
        lt_loans=lt_loans,
        cons_loans=_loan_array(),
        mort_loans=_loan_array(),
    )

    staged = market.schedule_firm_installments()
    principal_paid = market.settle_firm_installments(
        payable_principal_by_firm=np.array([17.5, 0.0]),
        payable_interest_by_firm=np.array([7.5, 0.0]),
        overwrite_bank_interest=True,
    )

    assert np.isclose(staged["scheduled_principal_due"][0], 35.0)
    assert np.isclose(staged["scheduled_interest_due"][0], 15.0)
    assert np.isclose(principal_paid[0], 17.5)
    assert np.isclose(market.compute_interest_paid_by_firm()[0], 7.5)
    assert np.isclose(market.compute_interest_received_by_bank()[0], 5.0)
    assert np.isclose(market.compute_interest_received_by_bank()[1], 2.5)
    assert np.isclose(market.states["st_loans"][0, 0, 0], 95.0)
    assert np.isclose(market.states["lt_loans"][0, 1, 0], 45.0)


def test_missed_firm_payment_capitalizes_interest_and_carries_principal_arrears():
    st_loans = _loan_array(n_borrowers=1)
    st_loans[0, 0, 0] = 100.0
    st_loans[1, 0, 0] = 0.10
    st_loans[2, 0, 0] = 30.0
    market = CreditMarket.from_data(
        country_name="TST",
        st_loans=st_loans,
        lt_loans=_loan_array(n_borrowers=1),
        cons_loans=_loan_array(n_borrowers=1),
        mort_loans=_loan_array(n_borrowers=1),
    )

    staged = market.schedule_firm_installments()
    market.settle_firm_installments(
        payable_principal_by_firm=np.zeros(1),
        payable_interest_by_firm=np.zeros(1),
        overwrite_bank_interest=True,
    )
    next_preview = market.compute_scheduled_firm_installments_preview()

    assert np.isclose(staged["contractual_interest_due"][0], 10.0)
    assert np.isclose(staged["contractual_principal_due"][0], 20.0)
    assert np.isclose(market.states["st_loans"][0, 0, 0], 110.0)
    assert np.isclose(next_preview["opening_interest_arrears"][0], 0.0)
    assert np.isclose(next_preview["opening_principal_arrears"][0], 20.0)
    assert np.isclose(next_preview["contractual_interest_due"][0], 11.0)
    assert np.isclose(next_preview["contractual_principal_due"][0], 19.0)
    assert np.isclose(next_preview["scheduled_interest_due"][0], 11.0)
    assert np.isclose(next_preview["scheduled_principal_due"][0], 39.0)


def test_capitalized_firm_interest_is_not_cash_interest_received():
    st_loans = _loan_array(n_borrowers=1)
    st_loans[0, 0, 0] = 100.0
    st_loans[1, 0, 0] = 0.10
    st_loans[2, 0, 0] = 30.0
    market = CreditMarket.from_data(
        country_name="TST",
        st_loans=st_loans,
        lt_loans=_loan_array(n_borrowers=1),
        cons_loans=_loan_array(n_borrowers=1),
        mort_loans=_loan_array(n_borrowers=1),
    )

    market.schedule_firm_installments()
    market.settle_firm_installments(
        payable_principal_by_firm=np.zeros(1),
        payable_interest_by_firm=np.zeros(1),
        overwrite_bank_interest=True,
    )

    assert np.isclose(market.states["st_loans"][0, 0, 0], 110.0)
    assert np.isclose(market.compute_interest_paid_by_firm()[0], 0.0)
    assert np.isclose(market.compute_interest_received_by_bank()[0], 0.0)


def test_principal_arrears_do_not_exceed_outstanding_principal_next_period():
    st_loans = _loan_array(n_borrowers=1)
    st_loans[0, 0, 0] = 30.0
    st_loans[1, 0, 0] = 0.10
    st_loans[2, 0, 0] = 30.0
    market = CreditMarket.from_data(
        country_name="TST",
        st_loans=st_loans,
        lt_loans=_loan_array(n_borrowers=1),
        cons_loans=_loan_array(n_borrowers=1),
        mort_loans=_loan_array(n_borrowers=1),
    )

    market.schedule_firm_installments()
    market.settle_firm_installments(
        payable_principal_by_firm=np.zeros(1),
        payable_interest_by_firm=np.zeros(1),
        overwrite_bank_interest=True,
    )
    next_preview = market.compute_scheduled_firm_installments_preview()

    assert np.isclose(market.states["st_loans"][0, 0, 0], 33.0)
    assert np.isclose(next_preview["opening_interest_arrears"][0], 0.0)
    assert np.isclose(next_preview["opening_principal_arrears"][0], 27.0)
    assert np.isclose(next_preview["contractual_principal_due"][0], 6.0)
    assert np.isclose(next_preview["scheduled_principal_due"][0], 33.0)


def test_firm_arrears_clear_when_loan_is_removed():
    st_loans = _loan_array(n_borrowers=1)
    st_loans[0, 0, 0] = 100.0
    st_loans[1, 0, 0] = 0.10
    st_loans[2, 0, 0] = 30.0
    market = CreditMarket.from_data(
        country_name="TST",
        st_loans=st_loans,
        lt_loans=_loan_array(n_borrowers=1),
        cons_loans=_loan_array(n_borrowers=1),
        mort_loans=_loan_array(n_borrowers=1),
    )

    market.schedule_firm_installments()
    market.settle_firm_installments(
        payable_principal_by_firm=np.zeros(1),
        payable_interest_by_firm=np.zeros(1),
        overwrite_bank_interest=True,
    )
    market.remove_loans_to_firm(np.array([0]))
    next_preview = market.compute_scheduled_firm_installments_preview()

    assert np.isclose(next_preview["opening_interest_arrears"][0], 0.0)
    assert np.isclose(next_preview["opening_principal_arrears"][0], 0.0)
    assert np.isclose(next_preview["scheduled_interest_due"][0], 0.0)
    assert np.isclose(next_preview["scheduled_principal_due"][0], 0.0)
