import numpy as np


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
            "consumption_loans_to_households",
            "mortgages_to_households",
            "total_outstanding_loans",
            "interest_received_on_loans",
            "interest_received_on_deposits",
            "interest_received",
            "firm_default_loan_writeoff",
            "firm_default_overdraft_writeoff",
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

    def test__compute_profits_subtracts_firm_default_credit_loss(self, test_banks):
        n_banks = test_banks.ts.current("n_banks")
        test_banks.ts.override_current("interest_received_on_loans", np.full(n_banks, 10.0))
        test_banks.ts.override_current("interest_received_on_deposits", np.full(n_banks, 2.0))
        test_banks.ts.override_current("firm_default_credit_loss", np.full(n_banks, 7.0))

        assert np.allclose(test_banks.compute_profits(), np.full(n_banks, 5.0))

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
