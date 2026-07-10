import numpy as np

from macromodel.markets.credit_market.credit_market import CreditMarket
from macromodel.markets.credit_market.func.clearing import DefaultCreditMarketClearer, PolednaCreditMarketClearer


class TestCreditMarket:
    def test__credit_market(self, test_credit_market):
        assert test_credit_market is not None


def _empty_loan_state(n_banks: int, n_borrowers: int) -> np.ndarray:
    return np.zeros((3, n_banks, n_borrowers))


def _configure_single_firm_credit_case(test_banks, test_firms, test_households) -> None:
    n_banks = test_banks.ts.current("n_banks")
    n_firms = test_firms.ts.current("n_firms")
    n_households = test_households.ts.current("n_households")
    first_firm = np.zeros(n_firms)
    first_firm[0] = 1.0

    test_banks.parameters.capital_adequacy_ratio = 0.1
    test_banks.parameters.enable_firm_loans_return_on_assets_restriction = False
    test_banks.parameters.enable_firm_loans_return_on_equity_restriction = False
    test_banks.parameters.enable_firm_loans_dscr_restriction = False
    test_banks.parameters.firm_loans_return_on_assets_ratio = -1.0
    test_banks.ts.override_current("equity", np.full(n_banks, 1.0e9))
    test_banks.ts.override_current("total_outstanding_loans", np.zeros(n_banks))
    test_banks.ts.override_current("interest_rates_on_short_term_firm_loans", np.full(n_banks, 0.05))
    test_banks.ts.override_current("interest_rates_on_long_term_firm_loans", np.full(n_banks, 0.05))

    test_firms.ts.override_current("target_short_term_credit", first_firm * 100.0)
    test_firms.ts.override_current("target_long_term_credit", np.zeros(n_firms))
    test_firms.ts.override_current("capital_inputs_stock_value", first_firm * 1.0e6)
    test_firms.ts.override_current("expected_capital_inputs_stock_value", first_firm * 1.0e6)
    test_firms.ts.override_current("expected_profits", np.zeros(n_firms))
    test_firms.ts.override_current("debt", np.zeros(n_firms))
    test_firms.ts.override_current("deposits", np.zeros(n_firms))

    test_households.ts.override_current("target_consumption_loans", np.zeros(n_households))
    test_households.ts.override_current("target_mortgage", np.zeros(n_households))


def _credit_market_with_clearer(clearer, test_banks, test_firms, test_households) -> CreditMarket:
    n_banks = test_banks.ts.current("n_banks")
    n_firms = test_firms.ts.current("n_firms")
    n_households = test_households.ts.current("n_households")
    market = CreditMarket.from_data(
        country_name="TST",
        st_loans=_empty_loan_state(n_banks, n_firms),
        lt_loans=_empty_loan_state(n_banks, n_firms),
        cons_loans=_empty_loan_state(n_banks, n_households),
        mort_loans=_empty_loan_state(n_banks, n_households),
    )
    market.functions["clearing"] = clearer
    return market


def test_default_clearer_integrates_with_credit_market_array_contract(test_banks, test_firms, test_households):
    _configure_single_firm_credit_case(test_banks, test_firms, test_households)
    market = _credit_market_with_clearer(
        DefaultCreditMarketClearer(
            allow_short_term_firm_loans=True,
            allow_household_loans=False,
            firms_max_number_of_banks_visiting=3,
            households_max_number_of_banks_visiting=3,
            consider_loan_type_fractions=False,
            credit_supply_temperature=1.0,
            interest_rates_selection_temperature=1.0,
            creditor_selection_is_deterministic=True,
            creditor_minimum_fill=False,
            debtor_minimum_fill=False,
        ),
        test_banks,
        test_firms,
        test_households,
    )

    market.clear(test_banks, test_firms, test_households, 0.0, 0.0, 0.0)

    assert np.isclose(test_firms.ts.current("received_short_term_credit")[0], 100.0)
    assert np.isclose(market.states["st_loans"][1, :, 0].max(), 0.05)
    assert market.states["st_loans"][2, :, 0].sum() > 0.0


def test_poledna_clearer_integrates_with_credit_market_array_contract(test_banks, test_firms, test_households):
    _configure_single_firm_credit_case(test_banks, test_firms, test_households)
    market = _credit_market_with_clearer(
        PolednaCreditMarketClearer(
            allow_short_term_firm_loans=True,
            allow_household_loans=False,
            firms_max_number_of_banks_visiting=3,
            households_max_number_of_banks_visiting=3,
            consider_loan_type_fractions=False,
            credit_supply_temperature=1.0,
            interest_rates_selection_temperature=1.0,
            creditor_selection_is_deterministic=True,
            creditor_minimum_fill=False,
            debtor_minimum_fill=False,
        ),
        test_banks,
        test_firms,
        test_households,
    )

    market.clear(test_banks, test_firms, test_households, 0.0, 0.0, 0.0)

    assert np.isclose(test_firms.ts.current("received_short_term_credit")[0], 100.0)
    assert np.isclose(market.states["st_loans"][1, :, 0].max(), 0.05)
    assert market.states["st_loans"][2, :, 0].sum() > 0.0


def test_resolver_enabled_clear_defers_consumer_loan_booking_to_settlement(
    test_banks, test_firms, test_households
):
    n_banks = test_banks.ts.current("n_banks")
    n_firms = test_firms.ts.current("n_firms")
    n_households = test_households.ts.current("n_households")
    test_households.configure_feasibility_resolver(True)
    test_banks.ts.override_current("equity", np.full(n_banks, 1.0e9))
    test_banks.ts.override_current("total_outstanding_loans", np.zeros(n_banks))
    test_banks.ts.override_current("interest_rates_on_household_consumption_loans", np.full(n_banks, 0.05))
    test_firms.ts.override_current("target_short_term_credit", np.zeros(n_firms))
    test_firms.ts.override_current("target_long_term_credit", np.zeros(n_firms))
    test_households.ts.override_current("target_consumption_loans", np.full(n_households, 100.0))
    test_households.ts.override_current("target_mortgage", np.zeros(n_households))
    market = _credit_market_with_clearer(
        DefaultCreditMarketClearer(
            allow_short_term_firm_loans=False,
            allow_household_loans=True,
            firms_max_number_of_banks_visiting=3,
            households_max_number_of_banks_visiting=3,
            consider_loan_type_fractions=False,
            credit_supply_temperature=1.0,
            interest_rates_selection_temperature=1.0,
            creditor_selection_is_deterministic=True,
            creditor_minimum_fill=False,
            debtor_minimum_fill=False,
        ),
        test_banks,
        test_firms,
        test_households,
    )

    market.clear(test_banks, test_firms, test_households, 0.0, 0.0, 0.0)

    granted = test_households.ts.current("received_consumption_loans")
    settlement = market.pending_granted_consumption_loans()
    assert granted.sum() > 0.0
    np.testing.assert_allclose(market.states["cons_loans"][0], np.zeros_like(market.states["cons_loans"][0]))
    np.testing.assert_allclose(settlement.sum(axis=0), granted)
    with np.testing.assert_raises_regex(RuntimeError, "must be booked before household loan servicing"):
        market.pay_household_installments()

    market.settle_granted_consumption_loans(
        credit_granted=granted,
        granted_consumer_credit_by_bank_and_household=settlement,
    )

    np.testing.assert_allclose(market.states["cons_loans"][0], settlement)


def test_granted_consumption_loan_settlement_reconciles_both_balance_sheet_sides(test_credit_market):
    test_credit_market._serviceable_loans_this_period["cons_loans"] = (
        test_credit_market.states["cons_loans"].copy()
    )
    settlement = np.zeros_like(test_credit_market.states["cons_loans"][0])
    settlement[0, 0] = 3.0
    if settlement.shape[0] > 1:
        settlement[1, 0] = 2.0
    if settlement.shape[1] > 1:
        settlement[0, 1] = 4.0
    opening_mortgages = test_credit_market.states["mort_loans"].copy()
    new_loans = np.zeros_like(test_credit_market.states["cons_loans"])
    new_loans[0] = settlement
    new_loans[1][settlement > 0.0] = 0.02
    new_loans[2][settlement > 0.0] = 1.0
    test_credit_market._pending_consumer_loans_this_period = new_loans
    granted = settlement.sum(axis=0)

    test_credit_market.settle_granted_consumption_loans(
        credit_granted=granted,
        granted_consumer_credit_by_bank_and_household=settlement,
    )

    np.testing.assert_allclose(test_credit_market.states["cons_loans"][0].sum(axis=0), granted)
    np.testing.assert_allclose(test_credit_market.states["cons_loans"][0].sum(axis=1), settlement.sum(axis=1))
    np.testing.assert_allclose(test_credit_market.states["mort_loans"], opening_mortgages)


def test_zero_granted_consumption_loan_settlement_books_nothing(test_credit_market):
    test_credit_market._serviceable_loans_this_period["cons_loans"] = (
        test_credit_market.states["cons_loans"].copy()
    )
    opening_consumption_loans = test_credit_market.states["cons_loans"].copy()
    new_loans = np.zeros_like(opening_consumption_loans)
    test_credit_market._pending_consumer_loans_this_period = new_loans

    test_credit_market.settle_granted_consumption_loans(
        credit_granted=np.zeros(opening_consumption_loans.shape[2]),
        granted_consumer_credit_by_bank_and_household=new_loans[0],
    )

    np.testing.assert_allclose(test_credit_market.states["cons_loans"], opening_consumption_loans)


def test_granted_consumption_loan_settlement_rejects_double_booking(test_credit_market):
    test_credit_market._serviceable_loans_this_period["cons_loans"] = (
        test_credit_market.states["cons_loans"].copy()
    )
    settlement = np.zeros_like(test_credit_market.states["cons_loans"][0])
    settlement[0, 0] = 5.0
    new_loans = np.zeros_like(test_credit_market.states["cons_loans"])
    new_loans[0] = settlement
    test_credit_market._pending_consumer_loans_this_period = new_loans
    test_credit_market.settle_granted_consumption_loans(
        credit_granted=settlement.sum(axis=0),
        granted_consumer_credit_by_bank_and_household=settlement,
    )

    with np.testing.assert_raises_regex(RuntimeError, "already been booked"):
        test_credit_market.settle_granted_consumption_loans(
            credit_granted=settlement.sum(axis=0),
            granted_consumer_credit_by_bank_and_household=settlement,
        )
