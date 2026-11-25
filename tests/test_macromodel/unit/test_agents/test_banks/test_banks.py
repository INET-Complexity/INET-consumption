import numpy as np
import pytest


class TestBanks:
    def test__banks_init(self, test_banks):
        assert set(test_banks.states.keys()) == {
            "corr_firms",
            "corr_households",
            "is_insolvent",
            "Firm Pass Through",
            "Firm ECT",
            "Household Consumption Pass Through",
            "Household Consumption ECT",
            "Household Mortgage Pass Through",
            "Household Mortgage ECT",
            "Firm Short Spread",
            "Firm Long Spread",
            "Household Consumption Spread",
            "Mortgage Spread",
        }

        ts_keys = [
            "n_banks",
            "equity",
            "deposits",
            "profits",
            "market_share",
            "deposits_from_firms",
            "deposits_from_households",
            "short_term_loans_to_firms",
            "long_term_loans_to_firms",
            "revolving_operating_facility_exposure",
            "consumption_loans_to_households",
            "mortgages_to_households",
            "total_outstanding_loans",
            "interest_received_on_loans",
            "interest_received_on_revolving_operating_facility",
            "interest_received_on_deposits",
            "interest_received",
            "firm_default_loan_writeoff",
            "firm_default_overdraft_writeoff",
            "firm_default_revolving_operating_facility_writeoff",
            "firm_default_credit_loss",
            "interest_rates_on_short_term_firm_loans",
            "interest_rates_on_long_term_firm_loans",
            "interest_rates_on_household_consumption_loans",
            "interest_rates_on_mortgages",
            "interest_rate_on_firm_deposits",
            "overdraft_rate_on_firm_deposits",
            "interest_rate_on_household_deposits",
            "overdraft_rate_on_household_deposits",
            # "interest_rate_on_government_debt",
        ]

        assert set(ts_keys).issubset(set(test_banks.ts.get_keys()))

    def test__update_loans_includes_revolving_operating_facility_exposure(self, test_banks, test_credit_market):
        n_banks = test_banks.ts.current("n_banks")
        facility_exposure = np.arange(1.0, n_banks + 1.0)
        cohort_loans = test_credit_market.compute_outstanding_loans_by_bank()

        test_banks.update_loans(
            credit_market=test_credit_market,
            revolving_operating_facility_exposure=facility_exposure,
        )

        np.testing.assert_allclose(
            test_banks.ts.current("revolving_operating_facility_exposure"),
            facility_exposure,
        )
        np.testing.assert_allclose(
            test_banks.ts.current("total_outstanding_loans"),
            cohort_loans + facility_exposure,
        )

    def test__update_deposits_books_each_depositor_gross_at_its_own_bank(self, test_banks):
        n_banks = test_banks.ts.current("n_banks")
        household_bank = np.array([0, 0, 1, 1, 0]) % n_banks
        household_lfa = np.array([100.0, -30.0, -5.0, 40.0, -1e-9])
        firm_bank = np.array([0, 1]) % n_banks
        firm_deposits = np.array([-7.0, 20.0])
        test_banks.update_deposits(
            current_firm_deposits=firm_deposits,
            current_household_deposits=household_lfa,
            firm_corresponding_bank=firm_bank,
            households_corresponding_bank=household_bank,
        )
        for name, ids, balances in (
            ("households", household_bank, household_lfa),
            ("firms", firm_bank, firm_deposits),
        ):
            deposits = test_banks.ts.current("deposits_from_" + name)
            overdrafts = test_banks.ts.current("overdrafts_to_" + name)
            for bank in range(n_banks):
                own = balances[ids == bank]
                assert deposits[bank] == own[own > 0.0].sum()
                assert overdrafts[bank] == -own[own < 0.0].sum()
            assert test_banks.ts.current("total_deposits_from_" + name)[0] == deposits.sum()
            assert test_banks.ts.current("total_overdrafts_to_" + name)[0] == overdrafts.sum()
        # The central-bank balance is unchanged by gross booking: liabilities less loans and overdrafts.
        np.testing.assert_allclose(
            test_banks.compute_deposits(),
            np.bincount(firm_bank, weights=firm_deposits, minlength=n_banks)
            + np.bincount(household_bank, weights=household_lfa, minlength=n_banks)
            + test_banks.ts.current("equity")
            - test_banks.ts.current("total_outstanding_loans"),
            rtol=1e-12,
        )

    def test__bank_receives_exactly_the_overdraft_interest_its_depositors_pay(self, test_banks):
        n_banks = test_banks.ts.current("n_banks")
        household_bank = np.array([0, 1, 0, 1]) % n_banks
        firm_bank = np.array([1, 0]) % n_banks
        test_banks.ts.override_current("deposits", np.zeros(n_banks))
        test_banks.record_household_deposit_interest(
            deposit_interest_received=np.array([1.0, 0.0, 2.0, 0.0]),
            overdraft_interest_paid=np.array([0.0, 3.0, 0.0, 4.0]),
            households_corresponding_bank=household_bank,
        )
        test_banks.record_firm_deposit_interest(
            deposit_interest_received=np.array([0.5, 0.0]),
            overdraft_interest_paid=np.array([0.0, 6.0]),
            firm_corresponding_bank=firm_bank,
        )
        expected = np.bincount(household_bank, weights=[-1.0, 3.0, -2.0, 4.0], minlength=n_banks) + np.bincount(
            firm_bank, weights=[-0.5, 6.0], minlength=n_banks
        )
        np.testing.assert_allclose(
            test_banks.compute_interest_received_on_deposits(central_bank_policy_rate=0.01), expected, rtol=1e-12
        )
        assert test_banks.ts.current("household_overdraft_interest_received").sum() == 7.0
        assert test_banks.ts.current("firm_overdraft_interest_received").sum() == 6.0

    def test__compute_profits_subtracts_firm_default_credit_loss(self, test_banks):
        n_banks = test_banks.ts.current("n_banks")
        test_banks.ts.override_current("interest_received_on_loans", np.full(n_banks, 10.0))
        test_banks.ts.override_current("interest_received_on_deposits", np.full(n_banks, 2.0))
        test_banks.ts.override_current("firm_default_credit_loss", np.full(n_banks, 7.0))

        assert np.allclose(test_banks.compute_profits(), np.full(n_banks, 5.0))

    def test__compute_profits_subtracts_consumer_principal_and_interest_losses(self, test_banks):
        n_banks = test_banks.ts.current("n_banks")
        test_banks.ts.override_current("interest_received_on_loans", np.full(n_banks, 10.0))
        test_banks.ts.override_current("interest_received_on_deposits", np.full(n_banks, 2.0))
        test_banks.ts.override_current("firm_default_credit_loss", np.zeros(n_banks))
        test_banks.ts.override_current("consumer_default_credit_loss", np.full(n_banks, 4.0))
        test_banks.ts.override_current("consumer_default_interest_income_loss", np.full(n_banks, 3.0))

        np.testing.assert_allclose(test_banks.compute_profits(), np.full(n_banks, 5.0))
        np.testing.assert_allclose(test_banks.compute_cash_distributable_profits(), np.full(n_banks, 5.0))

    def test__compute_equity_follows_profit_after_credit_loss(self, test_banks):
        n_banks = test_banks.ts.current("n_banks")
        test_banks.ts.override_current("equity", np.full(n_banks, 100.0))
        test_banks.ts.override_current("interest_received_on_loans", np.full(n_banks, 10.0))
        test_banks.ts.override_current("interest_received_on_deposits", np.zeros(n_banks))
        test_banks.ts.override_current("firm_default_credit_loss", np.full(n_banks, 6.0))
        test_banks.ts.override_current("profits", test_banks.compute_profits())

        assert np.allclose(test_banks.compute_equity(profit_taxes=0.0), np.full(n_banks, 104.0))

    def test__accrued_consumer_interest_is_accounting_profit_but_not_cash_distributable(self, test_banks):
        n_banks = test_banks.ts.current("n_banks")
        test_banks.ts.override_current("interest_received_on_loans", np.full(n_banks, 10.0))
        test_banks.ts.override_current("consumer_opening_interest_arrears_collected", np.full(n_banks, 3.0))
        test_banks.ts.override_current("consumer_interest_accrued", np.full(n_banks, 7.0))
        test_banks.ts.override_current("interest_received_on_deposits", np.zeros(n_banks))
        test_banks.ts.override_current("firm_default_credit_loss", np.zeros(n_banks))

        np.testing.assert_allclose(test_banks.compute_profits(), np.full(n_banks, 14.0))
        np.testing.assert_allclose(test_banks.compute_cash_distributable_profits(), np.full(n_banks, 7.0))

    # def test__banks_states(self, test_banks):
    #     assert test_banks is not None
    #     for state in [
    #         "corr_firms",
    #         "corr_households",
    #         "is_insolvent",
    #         "interest_rate_on_firm_short_term_loans_function",
    #         "interest_rate_on_firm_long_term_loans_function",
    #         "interest_rate_on_household_payday_loans_function",
    #         "interest_rate_on_household_consumption_expansion_loans_function",
    #         "interest_rate_on_mortgages_function",
    #     ]:
    #         assert state in test_banks.states.keys()
    #
    # def test__banks_ts(self, test_banks):
    #     for ts_key in [
    #         "n_banks",
    #         "equity",
    #         "deposits",
    #         "profits",
    #         "market_share",
    #         "deposits_from_firms",
    #         "deposits_from_households",
    #         "short_term_loans_to_firms",
    #         "long_term_loans_to_firms",
    #         "payday_loans_to_households",
    #         "consumption_expansion_loans_to_households",
    #         "mortgages_to_households",
    #         "total_outstanding_loans",
    #         "interest_received_on_loans",
    #         "interest_received_on_deposits",
    #         "interest_received",
    #         "interest_rates_on_short_term_firm_loans",
    #         "interest_rates_on_long_term_firm_loans",
    #         "interest_rates_on_household_payday_loans",
    #         "interest_rates_on_household_consumption_loans",
    #         "interest_rates_on_mortgages",
    #         "interest_rate_on_firm_deposits",
    #         "overdraft_rate_on_firm_deposits",
    #         "interest_rate_on_household_deposits",
    #         "overdraft_rate_on_household_deposits",
    #         # "interest_rate_on_government_debt",
    #     ]:
    #         assert ts_key in test_banks.ts.get_keys()
    #
    # def test__compute_profits(self, test_banks):
    #     assert test_banks.compute_profits()[0] == 80.0
    #
    # def test__compute_market_share(self, test_banks):
    #     assert test_banks.compute_market_share()[0] == 1.0
    #
    # def test__compute_equity(self, test_banks):
    #     assert test_banks.compute_equity(profit_taxes=0.25)[0] == 103.75
    #
    # def test__compute_deposits(self, test_banks):
    #     assert test_banks.compute_deposits()[0] == -60.0


def _two_bank_toy():
    """Two banks, each with overdrawn and positive depositors, booked by the native Banks methods."""
    from types import SimpleNamespace

    from macromodel.agents.banks.banks import Banks
    from macromodel.timeseries import TimeSeries

    zeros = np.zeros(2)
    fields = {
        name: zeros.copy()
        for name in (
            "deposits_from_firms",
            "deposits_from_households",
            "overdrafts_to_firms",
            "overdrafts_to_households",
            "household_overdraft_interest_received",
            "household_deposit_interest_paid",
            "household_overdraft_interest_accrued",
            "firm_overdraft_interest_received",
            "firm_deposit_interest_paid",
            "interest_received_on_loans",
            "consumer_opening_interest_arrears_collected",
            "mortgage_opening_interest_arrears_collected",
            "consumer_interest_accrued",
            "mortgage_interest_accrued",
            "interest_received_on_deposits",
            "firm_default_credit_loss",
            "consumer_default_credit_loss",
            "mortgage_default_credit_loss",
            "consumer_default_interest_income_loss",
            "household_sale_fee_income",
        )
    }
    totals = {
        "total_" + name: [0.0]
        for name in (
            "deposits_from_firms",
            "deposits_from_households",
            "overdrafts_to_firms",
            "overdrafts_to_households",
        )
    }
    banks = SimpleNamespace(
        ts=TimeSeries(
            n_banks=2,
            equity=np.array([50.0, 60.0]),
            total_outstanding_loans=np.array([100.0, 30.0]),
            deposits=np.array([10.0, -5.0]),
            **fields,
            **totals,
        )
    )
    for method in (
        "update_deposits",
        "record_household_deposit_interest",
        "record_firm_deposit_interest",
        "compute_interest_received_on_deposits",
        "compute_deposits",
        "compute_profits",
        "compute_cash_distributable_profits",
    ):
        setattr(banks, method, getattr(Banks, method).__get__(banks))
    return banks


def test__two_banks_book_mixed_sign_balances_gross_and_receive_each_depositors_interest():
    banks = _two_bank_toy()
    household_bank = np.array([0, 0, 0, 1, 1, 1])
    lfa = np.array([120.0, -30.0, -5.0, 40.0, -70.0, 10.0])
    firm_bank = np.array([0, 1, 1])
    firm_deposits = np.array([25.0, -8.0, 12.0])
    deposit_rate, overdraft_rate = np.array([0.01, 0.02]), np.array([0.05, 0.08])
    banks.update_deposits(
        current_firm_deposits=firm_deposits,
        current_household_deposits=lfa,
        firm_corresponding_bank=firm_bank,
        households_corresponding_bank=household_bank,
    )
    np.testing.assert_array_equal(banks.ts.current("deposits_from_households"), [120.0, 50.0])
    np.testing.assert_array_equal(banks.ts.current("overdrafts_to_households"), [35.0, 70.0])
    np.testing.assert_array_equal(banks.ts.current("deposits_from_firms"), [25.0, 12.0])
    np.testing.assert_array_equal(banks.ts.current("overdrafts_to_firms"), [0.0, 8.0])
    # Netting would have shown bank 1 households at -20 and hidden both deposits and overdrafts.
    household_deposit_interest = deposit_rate[household_bank] * np.maximum(lfa, 0.0)
    household_overdraft_interest = overdraft_rate[household_bank] * np.maximum(-lfa, 0.0)
    firm_deposit_interest = deposit_rate[firm_bank] * np.maximum(firm_deposits, 0.0)
    firm_overdraft_interest = overdraft_rate[firm_bank] * np.maximum(-firm_deposits, 0.0)
    # Interest left unpaid is accrued (held as consumer interest arrears), not received.
    accrued = np.array([0.0, 0.5, 0.0, 0.0, 2.0, 0.0])
    household_overdraft_interest = household_overdraft_interest - accrued
    banks.record_household_deposit_interest(
        deposit_interest_received=household_deposit_interest,
        overdraft_interest_paid=household_overdraft_interest,
        households_corresponding_bank=household_bank,
        overdraft_interest_accrued=accrued,
    )
    banks.record_firm_deposit_interest(
        deposit_interest_received=firm_deposit_interest,
        overdraft_interest_paid=firm_overdraft_interest,
        firm_corresponding_bank=firm_bank,
    )
    for bank in range(2):
        own, own_firms = household_bank == bank, firm_bank == bank
        assert banks.ts.current("household_overdraft_interest_received")[bank] == pytest.approx(
            household_overdraft_interest[own].sum(), rel=1e-15
        )
        assert banks.ts.current("household_deposit_interest_paid")[bank] == pytest.approx(
            household_deposit_interest[own].sum(), rel=1e-15
        )
        assert banks.ts.current("firm_overdraft_interest_received")[bank] == pytest.approx(
            firm_overdraft_interest[own_firms].sum(), rel=1e-15
        )
    policy_rate = 0.03
    received = banks.compute_interest_received_on_deposits(central_bank_policy_rate=policy_rate)
    expected = (
        policy_rate * np.array([10.0, 0.0])
        - policy_rate * np.array([0.0, 5.0])
        + np.bincount(household_bank, household_overdraft_interest - household_deposit_interest, minlength=2)
        + np.bincount(firm_bank, firm_overdraft_interest - firm_deposit_interest, minlength=2)
    )
    np.testing.assert_allclose(received, expected, rtol=1e-14)
    # Accrued overdraft interest is accounting profit but not distributable (or taxable) cash profit.
    banks.ts.override_current("interest_received_on_deposits", received)
    np.testing.assert_allclose(
        banks.compute_profits() - banks.compute_cash_distributable_profits(), [0.5, 2.0], rtol=1e-14
    )
    # The central-bank balance is liabilities less loans and overdrafts.
    np.testing.assert_allclose(
        banks.compute_deposits(),
        np.array([120.0 + 25.0 + 50.0 - 100.0 - 35.0, 50.0 + 12.0 + 60.0 - 30.0 - 70.0 - 8.0]),
        rtol=1e-14,
    )


def test__overdrafts_are_capital_requirement_exposure_and_count_in_insolvency(test_banks):
    from types import SimpleNamespace

    from macromodel.agents.banks.func.capital_requirement import capital_requirement_exposure
    from macromodel.agents.banks.func.demography import DefaultBankDemography

    n_banks = test_banks.ts.current("n_banks")
    loans, household_overdrafts, firm_overdrafts = (
        np.full(n_banks, 100.0),
        np.full(n_banks, 30.0),
        np.full(n_banks, 20.0),
    )
    test_banks.ts.override_current("total_outstanding_loans", loans)
    test_banks.ts.override_current("overdrafts_to_households", household_overdrafts)
    test_banks.ts.override_current("overdrafts_to_firms", firm_overdrafts)
    np.testing.assert_array_equal(capital_requirement_exposure(test_banks), np.full(n_banks, 150.0))
    # Loans stay loans only: the overdrafts are a separate exposure.
    np.testing.assert_array_equal(test_banks.ts.current("total_outstanding_loans"), loans)

    # Equity 6 covers 5% of the loans (5) but not of loans plus overdrafts (7.5).
    test_banks.functions["demography"] = DefaultBankDemography(solvency_ratio=0.05, reserve_weight=0.0)
    test_banks.ts.override_current("equity", np.full(n_banks, 6.0))
    test_banks.ts.override_current("deposits", np.zeros(n_banks))
    test_banks.states["is_insolvent"] = np.zeros(n_banks, dtype=bool)
    removed = []
    test_banks.handle_insolvency(credit_market=SimpleNamespace(remove_loans_by_bank=removed.append))
    assert removed == list(range(n_banks))
