"""Households plan on the receipts and obligations already known at planning.

Runs a short Simulation in legacy_stage5 and minimum_income modes, with a few
households pushed into overdraft, and checks the plan against the settled
period: dividends are planned on last period's declarations net of withholding,
and an opening overdraft is a senior use, financed by selling illiquid assets
if needed, with unpaid overdraft interest held as consumer interest arrears;
rental income is planned on the rent of the dwellings let after the housing market.
"""

import numpy as np
import pytest

from tests.test_macromodel.unit.test_country.test_minimum_income_entitlement import _production_simulation

PERIODS = 4
OVERDRAWN_FROM = 2


def _run(simulation):
    """Run, push every 40th household into overdraft at the start of period 2, record each period."""
    country = simulation.countries["FRA"]
    hh, banks, credit_market = country.households, country.banks, country.credit_market
    records = []
    overdrawn = np.zeros(hh.ts.current("n_households"), dtype=bool)
    booked = {}
    book = credit_market.book_unpaid_household_overdraft_interest

    def book_and_record(**kwargs):
        before = credit_market._consumer_interest_arrears_by_cell.copy()
        book(**kwargs)
        booked["arrears_added"] = credit_market._consumer_interest_arrears_by_cell - before

    credit_market.book_unpaid_household_overdraft_interest = book_and_record
    needs = []
    need = country._planned_overdraft_repayment_need

    def need_and_record(liquidity_shortfall):
        needs.append(need(liquidity_shortfall))
        return needs[-1]

    country._planned_overdraft_repayment_need = need_and_record
    planned_rent = {}
    realise = country.update_realised_metrics

    def realise_and_record(*args, **kwargs):
        planned_rent["income_rental"] = hh.ts.current("income_rental").copy()
        return realise(*args, **kwargs)

    country.update_realised_metrics = realise_and_record
    for period in range(PERIODS):
        if period == OVERDRAWN_FROM:
            overdrawn[::40] = True
            lfa = hh.ts.current("liquid_financial_assets")
            lfa[overdrawn] = -0.2 * np.abs(hh.ts.current("income")[overdrawn]) - 1.0
        needs.clear()
        opening = hh.ts.current("liquid_financial_assets").copy()
        opening_ifa = hh.ts.current("illiquid_financial_assets").copy()
        simulation.iterate(t=period)
        records.append(
            {
                "expected_dividends": hh.ts.current("expected_income_dividend_distributions").copy(),
                "dividends": hh.ts.current("income_dividend_distributions").copy(),
                "dividend_forecast_error": hh.ts.current("dividend_income_forecast_error").copy(),
                "firm_expired_distribution": country.firms.ts.current("dividend_fund_expired_distribution").copy(),
                "opening": opening,
                "opening_ifa": opening_ifa,
                "closing": hh.ts.current("liquid_financial_assets").copy(),
                "overdraft_repayment": hh.ts.current("overdraft_repayment").copy(),
                "liquidation_planned": hh.ts.current("liquidation_planned").copy(),
                "scheduled_overdraft_interest": hh.ts.current("scheduled_overdraft_interest").copy(),
                "unpaid_overdraft_interest": hh.ts.current("unpaid_overdraft_interest").copy(),
                "overdraft_interest_paid": hh.ts.current("overdraft_interest_paid").copy(),
                "financing_income_error": hh.ts.current("financing_income_error").copy(),
                "ledger_residual": hh.ts.current("stage5_cash_ledger_residual").copy(),
                "arrears_added": booked.pop("arrears_added"),
                "bank_overdraft_interest_accrued": banks.ts.current("household_overdraft_interest_accrued").copy(),
                "floor_support": hh.post_grant_feasible_plan.floor_support.copy(),
                "overdraft_repayment_need": needs[0],
                "planned_rental": planned_rent["income_rental"],
                "rental": hh.ts.current("income_rental").copy(),
                "rent_unpaid_absorbed": hh.ts.current("rent_unpaid_absorbed").copy(),
            }
        )
    return country, overdrawn, records


@pytest.fixture(scope="module", params=["legacy_stage5", "minimum_income"])
def run(request, datawrapper):
    return _run(_production_simulation(datawrapper, {"mode": request.param}))


def test_planned_dividends_equal_settled_dividends_when_no_cap_binds(run):
    _, _, records = run
    assert sum(record["dividends"].sum() for record in records) > 0.0
    for record in records:
        np.testing.assert_allclose(
            record["dividend_forecast_error"], record["dividends"] - record["expected_dividends"], rtol=0.0, atol=0.0
        )
        if record["firm_expired_distribution"].sum() == 0.0:
            np.testing.assert_allclose(record["expected_dividends"], record["dividends"], rtol=1e-12, atol=1e-9)
        else:
            # A binding payer cap leaves part of a declaration unpaid: only a negative error is possible.
            assert np.all(record["dividend_forecast_error"] <= 1e-9)


def test_cash_ledger_closes_every_period(run):
    _, _, records = run
    for record in records:
        np.testing.assert_allclose(record["ledger_residual"], 0.0, atol=1e-7)


def test_plan_repays_the_overdraft_when_resources_allow(run):
    _, overdrawn, records = run
    record = records[OVERDRAWN_FROM]
    overdraft = np.maximum(-record["opening"], 0.0)
    repaid = overdrawn & np.isclose(record["overdraft_repayment"], overdraft, rtol=1e-9, atol=0.0)
    assert repaid.sum() >= 10
    assert np.all(record["overdraft_repayment"] <= overdraft + 1e-6)
    # A plan that repays the whole overdraft leaves the account in credit unless realised income fell short.
    on_plan = repaid & (record["financing_income_error"] >= 0.0)
    assert np.all(record["closing"][on_plan] >= 0.0)
    # The overdraft never deepens on plan: unpaid interest is not charged to the account.
    on_plan = overdrawn & (record["financing_income_error"] >= 0.0)
    assert np.all(record["closing"][on_plan] >= record["opening"][on_plan] - 1e-6)


def test_support_never_finances_overdraft_repayment(run):
    _, _, records = run
    for record in records:
        # Support tops resources up to the floor and rent only, so a supported household repays nothing.
        supported = record["floor_support"] > 0.0
        np.testing.assert_array_equal(record["overdraft_repayment"][supported], 0.0)


def test_overdraft_repayment_is_financed_by_liquidation(run):
    _, overdrawn, records = run
    record = records[OVERDRAWN_FROM]
    need = record["overdraft_repayment_need"]
    assert np.all(need[~overdrawn & (record["opening"] >= 0.0)] == 0.0)
    selling = overdrawn & (need > 0.0) & (record["opening_ifa"] > 0.0)
    assert selling.sum() >= 10
    # The overdraft sale comes on top of the borrow-or-sell plan: total planned sale >= min(need, IFA).
    assert np.all(record["liquidation_planned"] >= np.minimum(need, record["opening_ifa"]) * (1.0 - 1e-12) - 1e-6)


def test_unpaid_overdraft_interest_is_consumer_interest_arrears_not_overdraft(run):
    country, _, records = run
    bank = country.households.states["Corresponding Bank ID"]
    n_banks = country.banks.ts.current("n_banks")
    assert sum(record["unpaid_overdraft_interest"].sum() for record in records) > 0.0
    for record in records:
        unpaid = record["unpaid_overdraft_interest"]
        expected = np.zeros_like(record["arrears_added"])
        expected[bank, np.arange(unpaid.size)] = unpaid
        np.testing.assert_allclose(record["arrears_added"], expected, rtol=1e-12, atol=1e-6)
        # The account is charged the interest paid; the bank accrues the rest, as for unpaid consumer interest.
        np.testing.assert_allclose(
            record["overdraft_interest_paid"], record["scheduled_overdraft_interest"] - unpaid, rtol=1e-12, atol=0.0
        )
        np.testing.assert_allclose(
            record["bank_overdraft_interest_accrued"], np.bincount(bank, weights=unpaid, minlength=n_banks), rtol=1e-12
        )


def test_rental_income_planned_after_the_housing_market_is_the_realised_rent(run):
    country, _, records = run
    tax = country.central_government.states["Income Tax"]
    assert sum(record["rental"].sum() for record in records) > 0.0
    for record in records:
        # Pass 2 plans the rent due on the dwellings let after clearing; realised rent is the rent received.
        np.testing.assert_allclose(
            record["planned_rental"],
            record["rental"] + (1.0 - tax) * record["rent_unpaid_absorbed"],
            rtol=1e-12,
            atol=1e-6,
        )
