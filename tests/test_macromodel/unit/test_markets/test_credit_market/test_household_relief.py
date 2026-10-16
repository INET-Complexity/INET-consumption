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
