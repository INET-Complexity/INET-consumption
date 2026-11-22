"""Independent arithmetic for current rates on lagged borrowing exposure."""

from dataclasses import replace

import numpy as np
import pytest

from macromodel.agents.households.borrowing_rates import (
    behavioural_borrowing_rate_inputs,
    lagged_annual_income_nominal,
)
from macromodel.agents.households.consumption_inputs import (
    IncomeHistories,
    MonetaryUnits,
    OpeningSnapshot,
    OpeningStockValues,
    PermanentIncomePayload,
    prepare_inputs,
)


def prepared_inputs(scale=1.0, current_income=99999.0, current_price=2.0, layout="aligned"):
    units = MonetaryUnits("EUR", 1)
    tags = dict(period_id=2, pass_kind="first", household_layout_key="rate-test")
    history = IncomeHistories(
        behavioural_income_nominal_history=np.array([[100, 200, 80], [200, 400, 160], [current_income] * 3]) * scale,
        guard_nominal_history=np.ones((3, 3)) * scale,
        consumer_price_history=[1, 2, current_price] if layout == "aligned" else [1, 2],
        current_consumer_price=current_price,
        initial_consumer_price=1.0,
        price_history_layout=layout,
        income_units=units,
        guard_units=units,
    )
    stocks = OpeningStockValues(
        *[
            np.array(v, dtype=float) * scale
            for v in (
                [200, -40, 0],
                [0, 0, 0],
                [0, 0, 0],
                [170, 250, 0],
                [90, 130, 0],
                [160, 240, 0],
                [80, 120, 0],
            )
        ]
    )
    snapshot = OpeningSnapshot(2, "rate-test", units, stocks)
    pi = PermanentIncomePayload(
        **tags, units=units, monetary_income_nominal=np.ones(3) * scale, pi_log_ratio_original=[0] * 3
    )
    return prepare_inputs(history, snapshot, pi, **tags)


def rates(prepared=None, **overrides):
    args = dict(
        household_bank_ids=np.array([0, 1, -1]),
        consumer_rates=np.array([0.01, 0.02]),
        overdraft_rates=np.array([0.03, 0.04]),
        mortgage_rates=np.array([0.04, 0.05]),
        expected_period_inflation=0.0,
    )
    return behavioural_borrowing_rate_inputs(prepared or prepared_inputs(), **(args | overrides))


@pytest.mark.parametrize("layout", ["aligned", "missing_current"])
def test_current_bank_rates_blend_and_lagged_nominal_exposures(layout):
    p = prepared_inputs(layout=layout)
    result = rates(p)
    # Prior real income is [100,200,80], valued at prior CPI=2 and annualised.
    np.testing.assert_allclose(result["lagged_annual_income_nominal"], [800, 1600, 640])
    np.testing.assert_allclose(result["consumer_rate_nominal"][:2], [0.02, 0.03])
    np.testing.assert_array_equal(result["consumer_exposure"], [80, 160, 0])
    np.testing.assert_allclose(
        result["real_borrowing_rate"],
        [
            (((1.02**4) - 1) * 80 + ((1.04**4) - 1) * 160) / 800,
            (((1.03**4) - 1) * 160 + ((1.05**4) - 1) * 240) / 1600,
            0,
        ],
    )
    # Principal, rather than the larger ledger debt including arrears, is used.
    np.testing.assert_array_equal(result["consumer_principal"], [80, 120, 0])


def test_current_income_and_prices_do_not_reweight_existing_exposure():
    reference = rates(prepared_inputs())
    changed = rates(prepared_inputs(current_income=1e12, current_price=10))
    np.testing.assert_array_equal(changed["real_borrowing_rate"], reference["real_borrowing_rate"])
    np.testing.assert_allclose(
        rates(prepared_inputs(scale=1e6))["real_borrowing_rate"],
        reference["real_borrowing_rate"],
        rtol=1e-12,
        atol=1e-14,
    )


def test_current_quote_moves_index_without_a_refinancing_event():
    before = rates()
    after = rates(consumer_rates=np.array([0.02, 0.02]))
    assert after["real_borrowing_rate"][0] > before["real_borrowing_rate"][0]
    np.testing.assert_array_equal(after["real_borrowing_rate"][1:], before["real_borrowing_rate"][1:])
    inflation = rates(expected_period_inflation=0.01)
    assert inflation["real_borrowing_rate"][0] == pytest.approx(
        (((1.02 / 1.01) ** 4 - 1) * 80 + ((1.04 / 1.01) ** 4 - 1) * 160) / 800
    )


def test_startup_floors_and_currency_units():
    p = prepared_inputs()
    h = replace(
        p.histories,
        behavioural_income_nominal_history=[[-1, 0, 1]],
        guard_nominal_history=[[10, 20, 30]],
        consumer_price_history=[1],
        current_consumer_price=1,
    )
    single = prepare_inputs(
        h,
        p.opening_snapshot,
        p.pi_payload,
        period_id=2,
        pass_kind="first",
        household_layout_key="rate-test",
    )
    np.testing.assert_allclose(lagged_annual_income_nominal(single), [40, 80, 120])


@pytest.mark.parametrize("ids", [[0.0, 1.0, -1.0], [-1, 1, -1], [0, 2, -1], [0]])
def test_invalid_or_missing_borrower_bank_fails(ids):
    with pytest.raises(ValueError, match="bank ID"):
        rates(household_bank_ids=np.asarray(ids))


@pytest.mark.parametrize("values", [[np.nan, 0.01], [-0.01, 0.01], [0.01]])
def test_invalid_rate_products_fail(values):
    with pytest.raises(ValueError, match="bank"):
        rates(overdraft_rates=np.asarray(values))


@pytest.mark.parametrize("current,previous", [(150, 100), (100, 150), (100, 100), (0, 100), (100, 0)])
def test_cashflow_equation_uses_quarterly_payment_change_and_preserves_other_components(current, previous):
    from macromodel.agents.households.cashflow_rates import DebtServicePayments
    from macromodel.agents.households.func.consumption import CreditAugmentedConsumption

    prepared = prepared_inputs()
    rule = CreditAugmentedConsumption(
        interest_rate_cashflow_propensity=-999,
        partial_adjustment_speed=0.56,
        real_borrowing_rate_propensity=-0.72,
    )
    kwargs = dict(lagged_real_consumption_budget=prepared.income_base_real, real_borrowing_rate=0.01)
    baseline, _ = rule.evaluate_prepared_target(prepared, **kwargs)
    payments = DebtServicePayments(np.full(3, current), np.full(3, previous))
    observed, _ = rule.evaluate_prepared_target(
        prepared,
        debt_service_payments=payments,
        consumer_debt_rate_delta=999,
        **kwargs,
    )
    deflator = prepared.histories.current_consumer_price / prepared.histories.initial_consumer_price
    expected = -0.75 * (current - previous) / (prepared.income_base_real * deflator)
    np.testing.assert_allclose(observed["target_consumption_debt_service_cashflow"], expected)
    np.testing.assert_allclose(
        observed["target_consumption_unguarded_delta_log_consumption"]
        - baseline["target_consumption_unguarded_delta_log_consumption"],
        expected,
        atol=1e-14,
    )
    # No lambda, annualisation, debt/income weight, or old rate term enters the replacement.
    for key in (
        "real_borrowing_rate",
        "real_income",
        "permanent_income",
        "wealth_contribution",
        "partial_adjustment_gap",
        "income_growth",
        "uncertainty",
        "real_lagged_consumption",
    ):
        np.testing.assert_array_equal(observed["target_consumption_" + key], baseline["target_consumption_" + key])
