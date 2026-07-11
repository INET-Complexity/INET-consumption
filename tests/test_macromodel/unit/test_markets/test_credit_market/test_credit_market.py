import numpy as np
import pytest

from macromodel.markets.credit_market.credit_market import CreditMarket
from macromodel.markets.credit_market.func.clearing import (
    DefaultCreditMarketClearer,
    PolednaCreditMarketClearer,
    _annuity_payment_factor,
)


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


def test_resolver_enabled_clear_defers_consumer_loan_booking_to_settlement(test_banks, test_firms, test_households):
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
        consumer_loan_maturity=test_banks.parameters.household_consumption_loan_maturity,
    )

    np.testing.assert_allclose(market.states["cons_loans"][0], settlement)


def test_granted_consumption_loan_settlement_reconciles_both_balance_sheet_sides(test_credit_market):
    test_credit_market._serviceable_loans_this_period["cons_loans"] = test_credit_market.states["cons_loans"].copy()
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
        consumer_loan_maturity=4,
    )

    np.testing.assert_allclose(test_credit_market.states["cons_loans"][0].sum(axis=0), granted)
    np.testing.assert_allclose(test_credit_market.states["cons_loans"][0].sum(axis=1), settlement.sum(axis=1))
    np.testing.assert_allclose(test_credit_market.states["mort_loans"], opening_mortgages)


def test_zero_granted_consumption_loan_settlement_books_nothing(test_credit_market):
    test_credit_market._serviceable_loans_this_period["cons_loans"] = test_credit_market.states["cons_loans"].copy()
    opening_consumption_loans = test_credit_market.states["cons_loans"].copy()
    new_loans = np.zeros_like(opening_consumption_loans)
    test_credit_market._pending_consumer_loans_this_period = new_loans

    test_credit_market.settle_granted_consumption_loans(
        credit_granted=np.zeros(opening_consumption_loans.shape[2]),
        granted_consumer_credit_by_bank_and_household=new_loans[0],
        consumer_loan_maturity=4,
    )

    np.testing.assert_allclose(test_credit_market.states["cons_loans"], opening_consumption_loans)


@pytest.mark.parametrize("consumer_loan_maturity", [0, float("nan"), 1.5])
def test_invalid_consumer_loan_maturity_rejects_settlement_without_booking(test_credit_market, consumer_loan_maturity):
    test_credit_market._serviceable_loans_this_period["cons_loans"] = test_credit_market.states["cons_loans"].copy()
    settlement = np.zeros_like(test_credit_market.states["cons_loans"][0])
    settlement[0, 0] = 5.0
    pending_loans = np.zeros_like(test_credit_market.states["cons_loans"])
    pending_loans[0] = settlement
    pending_loans[1][settlement > 0.0] = 0.02
    pending_loans[2][settlement > 0.0] = 1.0
    test_credit_market._pending_consumer_loans_this_period = pending_loans.copy()
    opening_loans = test_credit_market.states["cons_loans"].copy()

    with np.testing.assert_raises_regex(ValueError, "consumer_loan_maturity must be positive"):
        test_credit_market.settle_granted_consumption_loans(
            credit_granted=settlement.sum(axis=0),
            granted_consumer_credit_by_bank_and_household=settlement,
            consumer_loan_maturity=consumer_loan_maturity,
        )

    np.testing.assert_allclose(test_credit_market.states["cons_loans"], opening_loans)
    np.testing.assert_allclose(test_credit_market.pending_granted_consumption_loans(), settlement)


def test_granted_consumer_credit_remodulates_the_aggregate_household_schedule():
    cons_loans = _empty_loan_state(n_banks=2, n_borrowers=1)
    cons_loans[0, 0, 0] = 100.0
    cons_loans[1, 0, 0] = 0.02
    cons_loans[2, 0, 0] = 100.0 * _annuity_payment_factor(0.02, 8)
    opening_payment = cons_loans[2, 0, 0]
    market = CreditMarket.from_data(
        country_name="TST",
        st_loans=_empty_loan_state(n_banks=2, n_borrowers=1),
        lt_loans=_empty_loan_state(n_banks=2, n_borrowers=1),
        cons_loans=cons_loans,
        mort_loans=_empty_loan_state(n_banks=2, n_borrowers=1),
    )
    market._serviceable_loans_this_period["cons_loans"] = cons_loans.copy()
    settled_loans = _empty_loan_state(n_banks=2, n_borrowers=1)
    settled_loans[0, 1, 0] = 50.0
    settled_loans[1, 1, 0] = 0.04
    settled_loans[2, 1, 0] = 50.0 * _annuity_payment_factor(0.04, 8)
    market._pending_consumer_loans_this_period = settled_loans

    market.settle_granted_consumption_loans(
        credit_granted=np.array([50.0]),
        granted_consumer_credit_by_bank_and_household=settled_loans[0],
        consumer_loan_maturity=8,
    )
    principal_paid = market.pay_household_installments()

    expected_old_principal_paid = opening_payment - 100.0 * 0.02
    expected_principal = 150.0 - expected_old_principal_paid
    expected_schedule = expected_principal * _annuity_payment_factor(0.04, 8)
    np.testing.assert_allclose(principal_paid, np.array([expected_old_principal_paid]))
    np.testing.assert_allclose(market.states["cons_loans"][0, :, 0].sum(), expected_principal)
    np.testing.assert_allclose(market.states["cons_loans"][1, :, 0], np.array([0.04, 0.04]))
    np.testing.assert_allclose(
        market.states["cons_loans"][2, :, 0],
        expected_schedule * market.states["cons_loans"][0, :, 0] / expected_principal,
    )
    np.testing.assert_allclose(
        market.compute_scheduled_consumption_loan_payments_by_household(),
        np.array([expected_schedule]),
    )
    np.testing.assert_allclose(market.states["mort_loans"], np.zeros_like(market.states["mort_loans"]))


def test_granted_consumption_loan_settlement_rejects_double_booking(test_credit_market):
    test_credit_market._serviceable_loans_this_period["cons_loans"] = test_credit_market.states["cons_loans"].copy()
    settlement = np.zeros_like(test_credit_market.states["cons_loans"][0])
    settlement[0, 0] = 5.0
    new_loans = np.zeros_like(test_credit_market.states["cons_loans"])
    new_loans[0] = settlement
    test_credit_market._pending_consumer_loans_this_period = new_loans
    test_credit_market.settle_granted_consumption_loans(
        credit_granted=settlement.sum(axis=0),
        granted_consumer_credit_by_bank_and_household=settlement,
        consumer_loan_maturity=4,
    )

    with np.testing.assert_raises_regex(RuntimeError, "already been booked"):
        test_credit_market.settle_granted_consumption_loans(
            credit_granted=settlement.sum(axis=0),
            granted_consumer_credit_by_bank_and_household=settlement,
            consumer_loan_maturity=4,
        )


def test_household_service_snapshot_excludes_current_consumer_originations():
    cons_loans = _empty_loan_state(n_banks=2, n_borrowers=1)
    cons_loans[0, 0, 0] = 100.0
    cons_loans[1, 0, 0] = 0.02
    cons_loans[2, 0, 0] = 30.0
    market = CreditMarket.from_data(
        country_name="TST",
        st_loans=_empty_loan_state(2, 1),
        lt_loans=_empty_loan_state(2, 1),
        cons_loans=cons_loans,
        mort_loans=_empty_loan_state(2, 1),
    )
    new_loans = _empty_loan_state(2, 1)
    new_loans[0, 1, 0] = 50.0
    new_loans[1, 1, 0] = 0.04
    new_loans[2, 1, 0] = 10.0
    market._pending_consumer_loans_this_period = new_loans
    market.settle_granted_consumption_loans(
        credit_granted=np.array([50.0]),
        granted_consumer_credit_by_bank_and_household=new_loans[0],
        consumer_loan_maturity=8,
    )

    snapshot = market.prepare_household_service_snapshot()

    np.testing.assert_allclose(snapshot.consumer_total_due, np.array([30.0]))
    np.testing.assert_allclose(snapshot.newly_granted_consumer_loans, new_loans)
    with pytest.raises(ValueError):
        snapshot.consumer_total_due[0] = 0.0


def test_partial_consumer_payment_respects_interest_before_principal():
    cons_loans = _empty_loan_state(n_banks=2, n_borrowers=1)
    cons_loans[0, :, 0] = 100.0
    cons_loans[1, :, 0] = 0.10
    cons_loans[2, :, 0] = 60.0
    market = CreditMarket.from_data(
        country_name="TST",
        st_loans=_empty_loan_state(2, 1),
        lt_loans=_empty_loan_state(2, 1),
        cons_loans=cons_loans,
        mort_loans=_empty_loan_state(2, 1),
    )
    snapshot = market.prepare_household_service_snapshot()

    settlement = market.settle_consumer_payments(snapshot.consumer_total_due - 5.0)

    np.testing.assert_allclose(settlement.actual_payment, np.array([5.0]))
    np.testing.assert_allclose(settlement.interest_paid, np.array([5.0]))
    np.testing.assert_allclose(settlement.principal_paid, np.array([0.0]))
    np.testing.assert_allclose(market.states["cons_loans"][0, :, 0], np.array([100.0, 100.0]))
    np.testing.assert_allclose(market.compute_consumer_interest_accrued_by_bank(), np.array([7.5, 7.5]))


def test_consumer_arrears_reconcile_household_and_bank_assets_without_double_counting_principal():
    cons_loans = _empty_loan_state(n_banks=2, n_borrowers=1)
    cons_loans[0, :, 0] = np.array([80.0, 120.0])
    cons_loans[1, :, 0] = 0.10
    cons_loans[2, :, 0] = np.array([30.0, 45.0])
    market = CreditMarket.from_data(
        country_name="TST",
        st_loans=_empty_loan_state(2, 1),
        lt_loans=_empty_loan_state(2, 1),
        cons_loans=cons_loans,
        mort_loans=_empty_loan_state(2, 1),
    )
    snapshot = market.prepare_household_service_snapshot()
    market.settle_consumer_payments(snapshot.consumer_total_due)

    household_debt = market.compute_outstanding_consumption_loans_by_household()
    bank_assets = market.compute_outstanding_household_consumption_loans_by_bank()
    np.testing.assert_allclose(household_debt.sum(), bank_assets.sum())
    np.testing.assert_allclose(
        household_debt.sum(),
        market.states["cons_loans"][0].sum() + market._consumer_interest_arrears_by_cell.sum(),
    )
    assert np.all(market._consumer_principal_arrears_by_cell <= market.states["cons_loans"][0])


def test_consumer_arrears_carry_collect_once_and_remodulate_new_credit_next_period():
    cons_loans = _empty_loan_state(n_banks=2, n_borrowers=1)
    cons_loans[0, 0, 0] = 100.0
    cons_loans[1, 0, 0] = 0.10
    cons_loans[2, 0, 0] = 30.0
    market = CreditMarket.from_data(
        country_name="TST",
        st_loans=_empty_loan_state(2, 1),
        lt_loans=_empty_loan_state(2, 1),
        cons_loans=cons_loans,
        mort_loans=_empty_loan_state(2, 1),
    )

    first_snapshot = market.prepare_household_service_snapshot()
    market.settle_consumer_payments(first_snapshot.consumer_total_due)
    market.finalize_household_consumer_schedule()
    np.testing.assert_allclose(market._consumer_interest_arrears_by_cell[:, 0], np.array([10.0, 0.0]))
    np.testing.assert_allclose(market._consumer_principal_arrears_by_cell[:, 0], np.array([20.0, 0.0]))

    market._new_loans_this_period = {
        key: np.zeros_like(market.states[key]) for key in ("st_loans", "lt_loans", "cons_loans", "mort_loans")
    }
    market._serviceable_loans_this_period = {
        key: market.states[key].copy() for key in ("st_loans", "lt_loans", "cons_loans", "mort_loans")
    }
    market._household_service_snapshot = None
    market._consumer_payment_settlement = None
    market._last_interest_by_household.fill(0.0)
    market._last_interest_by_bank.fill(0.0)
    market._consumer_opening_arrears_collected_by_bank.fill(0.0)
    market._consumer_interest_accrued_by_bank.fill(0.0)

    new_loans = _empty_loan_state(2, 1)
    new_loans[0, 1, 0] = 50.0
    new_loans[1, 1, 0] = 0.04
    new_loans[2, 1, 0] = 10.0
    market._pending_consumer_loans_this_period = new_loans
    market.settle_granted_consumption_loans(
        credit_granted=np.array([50.0]),
        granted_consumer_credit_by_bank_and_household=new_loans[0],
        consumer_loan_maturity=8,
    )
    second_snapshot = market.prepare_household_service_snapshot()
    settlement = market.settle_consumer_payments(np.zeros(1))

    np.testing.assert_allclose(second_snapshot.consumer_total_due, np.array([60.0]))
    np.testing.assert_allclose(settlement.actual_payment, np.array([60.0]))
    np.testing.assert_allclose(
        market.compute_consumer_opening_interest_arrears_collected_by_bank(),
        np.array([10.0, 0.0]),
    )
    np.testing.assert_allclose(market.compute_consumer_interest_accrued_by_bank(), np.zeros(2))
    np.testing.assert_allclose(market.compute_recognized_interest_received_by_bank(), np.array([10.0, 0.0]))

    market.finalize_household_consumer_schedule()
    assert market.compute_scheduled_consumption_loan_payments_by_household()[0] > 0.0
    np.testing.assert_allclose(market._consumer_interest_arrears_by_cell, np.zeros((2, 1)))
    np.testing.assert_allclose(market._consumer_principal_arrears_by_cell, np.zeros((2, 1)))


def test_consumer_arrears_are_cleared_with_household_and_bank_loan_removal():
    cons_loans = _empty_loan_state(n_banks=2, n_borrowers=2)
    cons_loans[0] = np.array([[100.0, 20.0], [50.0, 30.0]])
    market = CreditMarket.from_data(
        country_name="TST",
        st_loans=_empty_loan_state(2, 1),
        lt_loans=_empty_loan_state(2, 1),
        cons_loans=cons_loans,
        mort_loans=_empty_loan_state(2, 2),
    )
    market._consumer_interest_arrears_by_cell[:] = np.array([[5.0, 2.0], [3.0, 4.0]])
    market._consumer_principal_arrears_by_cell[:] = np.array([[10.0, 1.0], [5.0, 2.0]])

    consumer_writeoff, _ = market.remove_loans_to_households(0)

    np.testing.assert_allclose(consumer_writeoff, 158.0)
    np.testing.assert_allclose(market._consumer_interest_arrears_by_cell[:, 0], np.zeros(2))
    np.testing.assert_allclose(market._consumer_principal_arrears_by_cell[:, 0], np.zeros(2))

    market.remove_loans_by_bank(0)
    np.testing.assert_allclose(market._consumer_interest_arrears_by_cell[0], np.zeros(2))
    np.testing.assert_allclose(market._consumer_principal_arrears_by_cell[0], np.zeros(2))


def test_household_service_snapshot_has_one_writer_per_period():
    market = CreditMarket.from_data(
        country_name="TST",
        st_loans=_empty_loan_state(1, 1),
        lt_loans=_empty_loan_state(1, 1),
        cons_loans=_empty_loan_state(1, 1),
        mort_loans=_empty_loan_state(1, 1),
    )
    market.prepare_household_service_snapshot()

    with pytest.raises(RuntimeError, match="already been snapshotted"):
        market.prepare_household_service_snapshot()
