import numpy as np
import pytest

from macromodel.markets.credit_market.credit_market import CreditMarket


def service_market(consumer_due=20.0, mortgage_due=10.0, mortgage_rate=0.0):
    consumer = np.zeros((3, 2, 1))
    mortgage = np.zeros_like(consumer)
    consumer[:, 0, 0] = [100, 0, consumer_due]
    mortgage[:, 0, 0] = [100, mortgage_rate, mortgage_due]
    return CreditMarket.from_data("TST", np.zeros((3, 2, 1)), np.zeros((3, 2, 1)), consumer, mortgage)


@pytest.mark.parametrize(
    "gap,consumer_relief,mortgage_relief,support",
    [(40.0, 20.0, 10.0, 10.0), (15.0, 15.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)],
)
def test_consumer_then_mortgage_relief_funds_only_residual_gap(gap, consumer_relief, mortgage_relief, support):
    market = service_market()
    market.prepare_household_service_snapshot()
    consumer = market.settle_consumer_payments(np.array([gap]))
    after_consumer = np.maximum(gap - consumer.unpaid_payment, 0.0)
    market.pay_household_mortgage_installments(after_consumer)
    assert consumer.unpaid_payment[0] == consumer_relief
    assert market._mortgage_unpaid_payment[0] == mortgage_relief
    assert max(after_consumer[0] - market._mortgage_unpaid_payment[0], 0.0) == support
    assert market._mortgage_principal_arrears_by_cell.sum() == mortgage_relief


def test_plan_example_resources_80_consumption_100_consumer_service_20():
    market = service_market(mortgage_due=0)
    market.prepare_household_service_snapshot()
    consumer = market.settle_consumer_payments(np.array([100 + 20 - 80.0]))
    residual = 40 - consumer.unpaid_payment[0]
    assert residual == 20  # previous code paid support 40 after suspending 20
    assert 80 + residual - 100 - consumer.actual_payment[0] == 0


def test_mortgage_arrears_match_bank_claims_collect_once_and_preserve_interest_income():
    market = service_market(mortgage_rate=0.1, mortgage_due=20.0)
    market.prepare_household_service_snapshot()
    market.settle_consumer_payments(np.array([40.0]))
    market.pay_household_mortgage_installments(np.array([20.0]))
    assert market._mortgage_interest_arrears_by_cell.sum() == 10
    assert market._mortgage_principal_arrears_by_cell.sum() == 10
    assert market.compute_outstanding_mortgages_by_household()[0] == 110
    assert market.compute_outstanding_mortgages_by_bank().sum() == 110
    assert market.compute_recognized_interest_received_by_bank().sum() == 10
    with pytest.raises(RuntimeError):
        market.pay_household_mortgage_installments()
    # Open the next period using the closing principal and arrears.
    market._household_service_snapshot = None
    market._consumer_payment_settlement = None
    market._mortgage_payments_settled = False
    market._mortgage_interest_paid.fill(0)
    market._last_interest_by_household.fill(0)
    market._last_interest_by_bank.fill(0)
    market.prepare_household_service_snapshot()
    market.settle_consumer_payments(np.zeros(1))
    market.pay_household_mortgage_installments(np.zeros(1))
    # Old interest 10 plus new contractual interest 9 are paid, but the old
    # accrued interest is not recognised as income a second time.
    assert market._mortgage_interest_paid[0] == pytest.approx(19)
    assert market.compute_recognized_interest_received_by_bank().sum() == pytest.approx(9)
    assert market._mortgage_interest_arrears_by_cell.sum() == 0
    assert market._mortgage_principal_arrears_by_cell.sum() == 0


def test_mortgage_accrual_and_collection_feed_bank_profit_without_cash_distribution(test_banks):
    banks = test_banks
    n = banks.ts.current("n_banks")
    original_profit = banks.compute_profits().copy()
    original_cash = banks.compute_cash_distributable_profits().copy()
    banks.ts.override_current("mortgage_interest_accrued", np.full(n, 10.0))
    banks.ts.override_current("mortgage_opening_interest_arrears_collected", np.full(n, 3.0))
    banks.ts.override_current("mortgage_default_credit_loss", np.full(n, 2.0))
    np.testing.assert_allclose(banks.compute_profits(), original_profit + 5)
    np.testing.assert_allclose(banks.compute_cash_distributable_profits(), original_cash - 5)


@pytest.mark.parametrize("opening_payment", [30.0, 110.0])
@pytest.mark.parametrize("new_bank", [0, 1])
def test_mortgage_settlement_preserves_surviving_contracts(opening_payment, new_bank):
    from macromodel.markets.credit_market.func.clearing import _annuity_payment_factor

    empty = np.zeros((3, 2, 1))
    opening = empty.copy()
    opening[:, 0, 0] = [100.0, 0.1, opening_payment]
    market = CreditMarket.from_data("TST", empty.copy(), empty.copy(), empty.copy(), opening)
    new = empty.copy()
    new_payment = 100.0 * _annuity_payment_factor(np.array([0.2]), 12)[0]
    new[:, new_bank, 0] = [100.0, 0.2, new_payment]
    market._add_new_loans("mort_loans", new)
    market.prepare_household_service_snapshot()
    market.settle_consumer_payments(np.zeros(1))
    market.pay_household_mortgage_installments(np.zeros(1))
    market.remove_repaid_loans(("mort_loans",))

    remaining_opening = 100.0 - (opening_payment - 10.0)
    expected_principal = new[0].copy()
    expected_principal[0, 0] += remaining_opening
    expected_interest = new[0] * new[1]
    expected_interest[0, 0] += remaining_opening * 0.1
    expected_rate = np.divide(
        expected_interest, expected_principal, out=np.zeros_like(expected_interest), where=expected_principal > 0
    )
    expected_payment = new[2].copy()
    if remaining_opening > 0:
        expected_payment[0, 0] += opening_payment
    np.testing.assert_allclose(market.states["mort_loans"][0], expected_principal)
    np.testing.assert_allclose(market.states["mort_loans"][1], expected_rate)
    np.testing.assert_allclose(market.states["mort_loans"][2], expected_payment)
    np.testing.assert_allclose(market.compute_outstanding_mortgages_by_bank(), expected_principal.sum(axis=1))

    # The next period must charge the surviving annuities, not the retired one.
    market._household_service_snapshot = None
    market._mortgage_payments_settled = False
    market.prepare_household_service_snapshot()
    assert market.current_household_service_snapshot().mortgage_total_due[0] == pytest.approx(expected_payment.sum())


def test_mortgage_interest_preview_matches_contract_after_actual_payment_relief():
    market = service_market(consumer_due=0.0, mortgage_rate=0.1, mortgage_due=20.0)
    market.prepare_household_service_snapshot()
    market.settle_consumer_payments(np.zeros(1))
    market.pay_household_mortgage_installments(np.array([20.0]))
    assert market._mortgage_principal_arrears_by_cell.sum() == 10.0
    assert market._mortgage_interest_arrears_by_cell.sum() == 10.0
    market._household_service_snapshot = None
    market._mortgage_payments_settled = False
    market.prepare_household_service_snapshot()
    expected = market.current_household_service_snapshot().mortgage_interest_by_cell.sum(axis=0)
    np.testing.assert_allclose(expected, 9.0)
    before = market.states["mort_loans"].copy()
    np.testing.assert_allclose(market.preview_opening_household_interest(), expected)
    np.testing.assert_array_equal(market.states["mort_loans"], before)


@pytest.mark.parametrize("shortfall", [5.0, 20.0, 110.0])
def test_new_mortgage_rate_excludes_nonaccruing_opening_principal_arrears(shortfall):
    from macromodel.markets.credit_market.func.clearing import _annuity_payment_factor

    empty = np.zeros((3, 1, 1))
    opening = empty.copy()
    opening[:, 0, 0] = [100.0, 0.1, 110.0]
    market = CreditMarket.from_data("TST", empty.copy(), empty.copy(), empty.copy(), opening)
    new = empty.copy()
    new[:, 0, 0] = [100.0, 0.2, 100.0 * _annuity_payment_factor(np.array([0.2]), 12)[0]]
    market._add_new_loans("mort_loans", new)
    market.prepare_household_service_snapshot()
    market.settle_consumer_payments(np.zeros(1))
    market.pay_household_mortgage_installments(np.array([shortfall]))
    assert market._mortgage_principal_arrears_by_cell.sum() == pytest.approx(min(shortfall, 100.0))
    # Every surviving opening-principal unit is in arrears; only the new
    # principal accrues next-period interest, at its own contractual rate.
    market._household_service_snapshot = None
    market._mortgage_payments_settled = False
    market.prepare_household_service_snapshot()
    assert market.current_household_service_snapshot().mortgage_interest_by_cell.sum() == pytest.approx(20.0)
    np.testing.assert_allclose(market.preview_opening_household_interest(), 20.0)
    # The suspended final principal is collected as arrears, not through an
    # additional opening annuity that would accelerate the new mortgage.
    expected_due = shortfall + new[2].sum()
    assert market.current_household_service_snapshot().mortgage_total_due[0] == pytest.approx(expected_due)
