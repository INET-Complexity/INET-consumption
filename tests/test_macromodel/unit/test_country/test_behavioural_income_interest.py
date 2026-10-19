"""Opening loan interest must survive servicing and refresh next period."""

from types import SimpleNamespace

import numpy as np
import pytest

from macromodel.country import Country
from macromodel.markets.credit_market.credit_market import CreditMarket
from macromodel.timeseries import TimeSeries


def _country_with_market(market):
    household_ts = TimeSeries(
        **{
            name: np.array([value])
            for name, value in {
                "expected_income_employee": 1000.0,
                "expected_income_social_transfers": 0.0,
                "expected_income_financial_assets": 0.0,
                "income_rental": 0.0,
            }.items()
        }
    )
    country = Country.__new__(Country)
    country._staged_household_income_shock = None
    country.credit_market = market
    country.households = SimpleNamespace(
        ts=household_ts,
        functions={"wealth": object()},
        compute_interest_paid_on_deposits=lambda **kwargs: np.zeros(1),
        current_consumption_imputed_rent=lambda: np.zeros(1),
    )
    country.banks = SimpleNamespace(
        ts=TimeSeries(
            interest_rate_on_household_deposits=np.zeros(1),
            overdraft_rate_on_household_deposits=np.zeros(1),
        )
    )
    country.economy = SimpleNamespace(ts=TimeSeries(subsistence_consumption=np.ones(1)))
    return country


@pytest.mark.parametrize("resolver, loan_key", [(False, "cons_loans"), (False, "mort_loans"), (True, "mort_loans")])
@pytest.mark.parametrize("case", ["amortising", "final", "new"])
def test_planned_interest_is_frozen_through_service_and_refreshed_next_period(resolver, loan_key, case):
    empty = np.zeros((3, 1, 1))
    opening_loan = empty.copy()
    if case != "new":
        opening_loan[:, 0, 0] = [100.0, 0.1, 110.0 if case == "final" else 30.0]
    market = CreditMarket.from_data(
        country_name="TST",
        st_loans=empty.copy(),
        lt_loans=empty.copy(),
        cons_loans=opening_loan.copy() if loan_key == "cons_loans" else empty.copy(),
        mort_loans=opening_loan.copy() if loan_key == "mort_loans" else empty.copy(),
    )
    country = _country_with_market(market)
    household_ts = country.households.ts
    country._record_behavioural_income(expected=True, initial=True)

    for period in range(2):
        expected_interest = (
            (0.0 if case == "new" else 10.0)
            if period == 0
            else {
                "amortising": 8.0,
                "final": 0.0,
                "new": 10.0,
            }[case]
        )
        # On period two, the previous snapshot still exists until clearing.
        country._record_behavioural_income(expected=True)
        np.testing.assert_allclose(
            household_ts.current("expected_behavioural_income_borrower_interest"), expected_interest
        )
        market._household_service_snapshot = None
        market._mortgage_payments_settled = False
        if case == "new" and period == 0:
            new_loan = empty.copy()
            new_loan[:, 0, 0] = [100.0, 0.1, 30.0]
            market.states[loan_key] = new_loan.copy()
            market._new_loans_this_period[loan_key] = new_loan.copy()
        if resolver:
            market.prepare_household_service_snapshot()
            market.pay_household_mortgage_installments()
        else:
            market.pay_household_installments()
        country._record_behavioural_income(expected=True, replace_current=True)
        np.testing.assert_allclose(
            household_ts.current("expected_behavioural_income_borrower_interest"), expected_interest
        )
        np.testing.assert_allclose(household_ts.current("expected_behavioural_income"), 1000.0 - expected_interest)
        assert len(household_ts.historic("expected_behavioural_income")) == period + 2


def test_planning_excludes_old_arrears_and_realised_income_uses_settled_interest():
    empty = np.zeros((3, 1, 1))
    consumer = empty.copy()
    consumer[:, 0, 0] = [50.0, 0.1, 20.0]
    market = CreditMarket.from_data(
        country_name="TST",
        st_loans=empty.copy(),
        lt_loans=empty.copy(),
        cons_loans=consumer,
        mort_loans=empty.copy(),
    )
    market._consumer_principal_arrears_by_cell[:] = 10.0
    market._consumer_interest_arrears_by_cell[:] = 7.0
    country = _country_with_market(market)
    hh = country.households
    country._record_behavioural_income(expected=True, initial=True)
    country._record_behavioural_income(expected=True)
    # Only current interest on performing principal: (50 - 10) * .1.
    np.testing.assert_allclose(hh.ts.current("expected_behavioural_income_loan_interest"), 4.0)
    market._consumer_principal_arrears_by_cell[:] = 0.0
    hh.compute_interest_paid_on_deposits = lambda **kwargs: np.array([3.0])
    country._record_behavioural_income(expected=True, replace_current=True)
    np.testing.assert_allclose(hh.ts.current("expected_behavioural_income_borrower_interest"), 7.0)
    # Realised reporting uses paid interest, including collected old interest.
    hh.ts["income_employee"] = np.array([1000.0])
    hh.ts["income_social_transfers"] = np.zeros(1)
    hh.ts["income_financial_assets"] = np.zeros(1)
    country._record_behavioural_income(expected=False, initial=True)
    hh.ts["interest_paid_on_loans"] = np.array([11.0])
    hh.ts["interest_paid_on_deposits"] = np.array([2.0])
    country._record_behavioural_income(expected=False)
    np.testing.assert_allclose(hh.ts.current("behavioural_income_borrower_interest"), 13.0)
    np.testing.assert_allclose(hh.ts.current("behavioural_income"), 987.0)
    # Stored planning interest is unaffected by realised reporting.
    np.testing.assert_allclose(hh.ts.current("expected_behavioural_income_loan_interest"), 4.0)
