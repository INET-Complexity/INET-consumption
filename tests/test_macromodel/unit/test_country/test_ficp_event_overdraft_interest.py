"""Unpaid overdraft interest booked after the Stage 6 FICP event snapshot must reach the event."""

from types import SimpleNamespace

import numpy as np
import pytest

import macromodel.country.country as country_module
from macromodel.agents.households.households import Households
from macromodel.country import Country
from macromodel.timeseries import TimeSeries

BOOKED_INTEREST = 47.5


def _country_with_new_event_and_zero_interest_snapshot():
    cons_loans = np.zeros((3, 2, 2))
    cons_loans[0] = np.array([[60.0, 20.0], [40.0, 30.0]])
    credit_market = country_module.CreditMarket.from_data(
        country_name="TST",
        st_loans=np.zeros((3, 2, 1)),
        lt_loans=np.zeros((3, 2, 1)),
        cons_loans=cons_loans,
        mort_loans=np.zeros((3, 2, 2)),
    )
    credit_market._consumer_principal_arrears_by_cell[:] = np.array([[10.0, 1.0], [5.0, 2.0]])
    households_ts = TimeSeries(
        n_households=2,
        ficp_forgiveness_event=np.array([True, False]),
        ficp_forgiveness_event_emitted=np.array([True, False]),
        ficp_forgiveness_event_processed=np.array([False, False]),
        ficp_forgiveness_event_stage=np.array([0.0, 0.0]),
        ficp_forgiveness_event_episode_id=np.array([1.0, 0.0]),
        ficp_forgiveness_event_horizon_end_period=np.array([56.0, 0.0]),
        ficp_forgiveness_event_residual_contractual_principal=np.array([85.0, 0.0]),
        ficp_forgiveness_event_residual_principal_arrears=np.array([15.0, 0.0]),
        # Stage 6 snapshot: taken before the overdraft interest is booked.
        ficp_forgiveness_event_residual_interest_arrears=np.array([0.0, 0.0]),
    )
    banks_ts = TimeSeries(
        n_banks=2,
        consumer_default_loan_writeoff=np.zeros(2),
        total_consumer_default_loan_writeoff=[0.0],
        consumer_default_principal_arrears=np.zeros(2),
        total_consumer_default_principal_arrears=[0.0],
        consumer_default_credit_loss=np.zeros(2),
        total_consumer_default_credit_loss=[0.0],
        consumer_default_interest_income_loss=np.zeros(2),
        total_consumer_default_interest_income_loss=[0.0],
        consumer_default_npl=np.zeros(2),
        consumer_default_npl_denominator=np.zeros(2),
    )
    return SimpleNamespace(
        households=SimpleNamespace(ts=households_ts),
        banks=SimpleNamespace(ts=banks_ts),
        credit_market=credit_market,
    )


def _book_unpaid_overdraft_interest(country, *, refresh_event: bool):
    """Mirror the Stage 5 order: book unpaid overdraft interest, then refresh new events."""
    credit_market = country.credit_market
    credit_market.book_unpaid_household_overdraft_interest(
        unpaid=np.array([BOOKED_INTEREST, 0.0]), corresponding_bank=np.array([1, 0])
    )
    if refresh_event:
        Households.refresh_new_ficp_event_interest_arrears(
            country.households, credit_market.current_consumer_debt_components_by_household()[2]
        )


def test_event_without_refresh_does_not_reconcile_with_booked_overdraft_interest():
    country = _country_with_new_event_and_zero_interest_snapshot()
    _book_unpaid_overdraft_interest(country, refresh_event=False)
    with pytest.raises(RuntimeError, match="does not reconcile with the live consumer-loan state"):
        Country._process_ficp_forgiveness_events(country)


def test_refreshed_event_reconciles_and_write_off_includes_booked_interest():
    country = _country_with_new_event_and_zero_interest_snapshot()
    _book_unpaid_overdraft_interest(country, refresh_event=True)

    np.testing.assert_allclose(
        country.households.ts.current("ficp_forgiveness_event_residual_interest_arrears"), [BOOKED_INTEREST, 0.0]
    )
    # Principal fields are untouched.
    np.testing.assert_allclose(
        country.households.ts.current("ficp_forgiveness_event_residual_principal_arrears"), [15.0, 0.0]
    )

    _, pending_events = Country._process_ficp_forgiveness_events(country)

    assert pending_events == ((0, 1),)
    # Household 0's interest sat in bank 1's cell.
    np.testing.assert_allclose(
        country.banks.ts.current("consumer_default_interest_income_loss"), [0.0, BOOKED_INTEREST]
    )


def test_refresh_leaves_events_from_earlier_periods_and_other_households_unchanged():
    country = _country_with_new_event_and_zero_interest_snapshot()
    country.households.ts.override_current("ficp_forgiveness_event_stage", np.array([1.0, 0.0]))
    Households.refresh_new_ficp_event_interest_arrears(country.households, np.array([5.0, 6.0]))
    np.testing.assert_allclose(
        country.households.ts.current("ficp_forgiveness_event_residual_interest_arrears"), [0.0, 0.0]
    )
