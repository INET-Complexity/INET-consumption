"""Overdrafts are an explicit bank instrument.

A household's negative liquid balance is an overdraft loan of its own bank. Each
period the bank books deposits and overdrafts gross, receives exactly the
overdraft interest its depositors pay, and the households plan that interest as
debt service. Runs a short Simulation in legacy_stage5 and minimum_income modes
with a few households pushed into overdraft.
"""

import numpy as np
import pytest

from tests.test_macromodel.unit.test_country.test_minimum_income_entitlement import _production_simulation

PERIODS = 4
OVERDRAWN_FROM = 2


def _run(simulation):
    """Run, push every 40th household into overdraft at the start of period 2, record each period."""
    country = simulation.countries["FRA"]
    hh, banks = country.households, country.banks
    records = []
    overdrawn = np.zeros(hh.ts.current("n_households"), dtype=bool)
    for period in range(PERIODS):
        if period == OVERDRAWN_FROM:
            overdrawn[::40] = True
            lfa = hh.ts.current("liquid_financial_assets")
            lfa[overdrawn] = -0.2 * np.abs(hh.ts.current("income")[overdrawn]) - 1.0
        opening = hh.ts.current("liquid_financial_assets").copy()
        opening_equity = banks.ts.current("equity").copy()
        simulation.iterate(t=period)
        bank = {
            name: np.asarray(banks.ts.current(name), dtype=float).copy()
            for name in (
                "equity",
                "profits",
                "cash_distributable_profits",
                "dividend_fund_settlement_debit",
                "deposits",
                "deposits_from_firms",
                "deposits_from_households",
                "overdrafts_to_firms",
                "overdrafts_to_households",
                "total_outstanding_loans",
                "consumer_interest_accrued",
                "mortgage_interest_accrued",
                "household_overdraft_interest_accrued",
            )
        }
        bank["opening_equity"] = opening_equity
        bank["profit_tax"] = country.central_government.states["Profit Tax"]
        bank["insolvency_rate"] = country.economy.ts.current("bank_insolvency_rate")[0]
        records.append(
            {
                "opening": opening,
                "closing": hh.ts.current("liquid_financial_assets").copy(),
                "overdraft_interest_paid": hh.ts.current("overdraft_interest_paid").copy(),
                "scheduled_overdraft_interest": hh.ts.current("scheduled_overdraft_interest").copy(),
                "unpaid_overdraft_interest": hh.ts.current("unpaid_overdraft_interest").copy(),
                "bank_overdraft_interest": banks.ts.current("household_overdraft_interest_received").copy(),
                "bank_deposit_interest": banks.ts.current("household_deposit_interest_paid").copy(),
                "deposit_rate": banks.ts.current("interest_rate_on_household_deposits").copy(),
                "bank_deposits": banks.ts.current("deposits_from_households").copy(),
                "bank_overdrafts": banks.ts.current("overdrafts_to_households").copy(),
                "ledger_residual": hh.ts.current("stage5_cash_ledger_residual").copy(),
                "early_repayment": hh.ts.current("early_consumer_repayment").copy(),
                "support": hh.ts.current("stage5_subsistence_support").copy(),
                "bank": bank,
            }
        )
    return country, overdrawn, records


@pytest.fixture(scope="module", params=["legacy_stage5", "minimum_income"])
def run(request, datawrapper):
    return _run(_production_simulation(datawrapper, {"mode": request.param}))


def test_bank_receives_exactly_the_overdraft_interest_its_households_pay(run):
    country, _, records = run
    banks = country.households.states["Corresponding Bank ID"]
    n_banks = country.banks.ts.current("n_banks")
    assert records[OVERDRAWN_FROM]["overdraft_interest_paid"].sum() > 0.0
    for record in records:
        paid = record["overdraft_interest_paid"]
        np.testing.assert_array_equal(paid >= 0.0, True)
        for bank in range(n_banks):
            np.testing.assert_allclose(
                record["bank_overdraft_interest"][bank], paid[banks == bank].sum(), rtol=1e-12, atol=0.0
            )
        np.testing.assert_allclose(record["bank_overdraft_interest"].sum(), paid.sum(), rtol=1e-12, atol=0.0)
        # Interest is charged on the opening balance at the household's own bank, and on negative balances only.
        np.testing.assert_array_equal(paid[record["opening"] >= 0.0], 0.0)
        np.testing.assert_allclose(
            record["bank_deposit_interest"],
            np.bincount(
                banks, weights=record["deposit_rate"][banks] * np.maximum(record["opening"], 0.0), minlength=n_banks
            ),
            rtol=1e-12,
        )


def test_banks_book_household_balances_gross(run):
    country, _, records = run
    banks = country.households.states["Corresponding Bank ID"]
    n_banks = country.banks.ts.current("n_banks")
    for record in records:
        closing = record["closing"]
        np.testing.assert_array_equal(
            record["bank_deposits"], np.bincount(banks, weights=np.maximum(closing, 0.0), minlength=n_banks)
        )
        np.testing.assert_array_equal(
            record["bank_overdrafts"], np.bincount(banks, weights=np.maximum(-closing, 0.0), minlength=n_banks)
        )
    assert records[OVERDRAWN_FROM]["bank_overdrafts"].sum() > 0.0


def test_planned_overdraft_interest_is_the_interest_charged(run):
    _, overdrawn, records = run
    record = records[OVERDRAWN_FROM]
    assert np.all(record["scheduled_overdraft_interest"][overdrawn] > 0.0)
    unpaid = record["unpaid_overdraft_interest"]
    assert np.all((unpaid >= 0.0) & (unpaid <= record["scheduled_overdraft_interest"]))
    # The account is charged the interest the plan pays; the unpaid part becomes consumer interest arrears.
    np.testing.assert_allclose(
        record["overdraft_interest_paid"], record["scheduled_overdraft_interest"] - unpaid, rtol=1e-12, atol=0.0
    )


def test_overdrawn_households_repay_the_overdraft_before_anything_else(run):
    _, overdrawn, records = run
    record = records[OVERDRAWN_FROM]
    # No early consumer repayment while the account stays overdrawn.
    still_overdrawn = record["closing"] < 0.0
    np.testing.assert_array_equal(record["early_repayment"][still_overdrawn], 0.0)
    # Paying the planned interest, the overdraft does not deepen unless the interest went unpaid.
    deepened = record["closing"] < record["opening"] - record["unpaid_overdraft_interest"] - 1e-6
    assert not np.any(deepened & overdrawn & (record["unpaid_overdraft_interest"] == 0.0))


def test_cash_ledger_closes_and_plans_leave_no_negative_residue(run):
    _, overdrawn, records = run
    for record in records:
        np.testing.assert_allclose(record["ledger_residual"], 0.0, atol=1e-7)
        # The plan rounds its resources down, so a plan that spends everything (including support-pinned
        # and unmet-floor households) closes at a small positive balance, not at a residue below zero.
        residue = (record["closing"] > -1e-6) & (record["closing"] < 0.0)
        assert residue.sum() == 0, f"{residue.sum()} households closed in (-1e-6, 0)"


def test_bank_equity_flow_and_balance_sheet_identities_hold_every_period(run):
    country, _, records = run
    banks = country.households.states["Corresponding Bank ID"]
    n_banks = country.banks.ts.current("n_banks")
    for record in records:
        bank = record["bank"]
        assert bank["insolvency_rate"] == 0.0
        # Equity moves by profit less tax on distributable profit less the dividend-fund debit.
        np.testing.assert_allclose(
            bank["equity"],
            bank["opening_equity"]
            + bank["profits"]
            - bank["profit_tax"] * np.maximum(0.0, bank["cash_distributable_profits"])
            - bank["dividend_fund_settlement_debit"],
            rtol=1e-12,
        )
        # Accrued interest, including unpaid overdraft interest, is profit but not distributable cash.
        np.testing.assert_allclose(
            bank["profits"] - bank["cash_distributable_profits"],
            bank["consumer_interest_accrued"]
            + bank["mortgage_interest_accrued"]
            + bank["household_overdraft_interest_accrued"],
            rtol=1e-9,
            atol=1e-3,
        )
        np.testing.assert_allclose(
            bank["household_overdraft_interest_accrued"],
            np.bincount(banks, weights=record["unpaid_overdraft_interest"], minlength=n_banks),
            rtol=1e-12,
        )
        # Balance sheet: central-bank balance = deposits + equity - loans - overdrafts.
        np.testing.assert_allclose(
            bank["deposits"],
            bank["deposits_from_firms"]
            + bank["deposits_from_households"]
            + bank["equity"]
            - bank["total_outstanding_loans"]
            - bank["overdrafts_to_firms"]
            - bank["overdrafts_to_households"],
            rtol=1e-12,
        )
