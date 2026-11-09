"""Increment (m): the mortgage secured on a sold property is repaid from the proceeds."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from macromodel.country.country import Country
from macromodel.markets.credit_market.credit_market import CreditMarket
from macromodel.markets.housing_market.sales import sold_collateral_share


def _mortgage_market(principal=(100.0,), instalment=(10.0,), rate=0.0) -> CreditMarket:
    """One household; one cell per bank with zero-rate annuity terms."""
    n_banks = len(principal)
    mortgage = np.zeros((3, n_banks, 1))
    mortgage[0, :, 0] = principal
    mortgage[1, :, 0] = rate
    mortgage[2, :, 0] = instalment
    empty = np.zeros((3, n_banks, 1))
    market = CreditMarket.from_data("TST", empty.copy(), empty.copy(), empty.copy(), mortgage)
    market.prepare_household_service_snapshot()
    market.settle_consumer_payments(np.zeros(1))
    return market


def test_principal_after_scheduled_service_excludes_this_period_principal():
    market = _mortgage_market()
    np.testing.assert_allclose(market.mortgage_principal_after_scheduled_service(), [90.0])


def test_owner_selling_only_property_repays_the_whole_mortgage():
    market = _mortgage_market()
    market.pay_household_mortgage_installments(np.zeros(1), sale_repayment=np.array([90.0]))

    loans = market.states["mort_loans"]
    assert loans[0].sum() == pytest.approx(0.0)
    assert loans[2].sum() == pytest.approx(0.0)
    # The repayment is financing, not scheduled service.
    assert market.compute_mortgage_principal_paid_by_household()[0] == pytest.approx(10.0)
    assert market.compute_outstanding_mortgages_by_bank().sum() == pytest.approx(0.0)


def test_half_collateral_sale_halves_principal_and_instalment_keeping_maturity():
    market = _mortgage_market()
    market.pay_household_mortgage_installments(np.zeros(1), sale_repayment=np.array([45.0]))

    loans = market.states["mort_loans"]
    assert loans[0, 0, 0] == pytest.approx(45.0)
    assert loans[2, 0, 0] == pytest.approx(5.0)  # 45 / 5 = 9 periods left, as with 90 / 10


def test_zero_repayment_leaves_settlement_unchanged():
    reference = _mortgage_market(rate=0.01)
    reference.pay_household_mortgage_installments(np.array([3.0]))
    repaid = _mortgage_market(rate=0.01)
    repaid.pay_household_mortgage_installments(np.array([3.0]), sale_repayment=np.zeros(1))

    np.testing.assert_array_equal(repaid.states["mort_loans"], reference.states["mort_loans"])
    np.testing.assert_array_equal(
        repaid._mortgage_principal_arrears_by_cell, reference._mortgage_principal_arrears_by_cell
    )


def test_mover_new_mortgage_is_untouched_by_the_old_loan_repayment():
    market = _mortgage_market()
    new_loan = np.zeros_like(market.states["mort_loans"])
    new_loan[:, 0, 0] = [200.0, 0.01, 5.0]
    market._add_new_loans("mort_loans", new_loan)

    market.pay_household_mortgage_installments(np.zeros(1), sale_repayment=np.array([90.0]))

    loans = market.states["mort_loans"]
    assert loans[0, 0, 0] == pytest.approx(200.0)
    assert loans[1, 0, 0] == pytest.approx(0.01)
    assert loans[2, 0, 0] == pytest.approx(5.0)


def test_repayment_clears_principal_arrears_first():
    market = _mortgage_market()
    # Relief on the whole instalment: 10 of principal goes into arrears.
    market.pay_household_mortgage_installments(np.array([10.0]), sale_repayment=np.array([90.0]))

    assert market._mortgage_principal_arrears_by_cell.sum() == pytest.approx(0.0)
    assert market.states["mort_loans"][0, 0, 0] == pytest.approx(10.0)


def test_repayment_is_split_pro_rata_across_banks():
    market = _mortgage_market(principal=(60.0, 40.0), instalment=(6.0, 4.0))
    np.testing.assert_allclose(market.mortgage_principal_after_scheduled_service(), [90.0])

    market.pay_household_mortgage_installments(np.zeros(1), sale_repayment=np.array([45.0]))

    loans = market.states["mort_loans"]
    np.testing.assert_allclose(loans[0, :, 0], [27.0, 18.0])
    np.testing.assert_allclose(loans[2, :, 0], [3.0, 2.0])
    assert market.compute_outstanding_mortgages_by_bank().sum() == pytest.approx(45.0)


def test_repayment_above_remaining_principal_is_rejected():
    market = _mortgage_market()
    with pytest.raises(RuntimeError, match="exceeds the opening principal"):
        market.pay_household_mortgage_installments(np.zeros(1), sale_repayment=np.array([95.0]))


def _sales(rows):
    return pd.DataFrame(
        {
            "sales_types": ["Sell"] * len(rows),
            "property_id": [r[0] for r in rows],
            "buyer_id": [r[1] for r in rows],
            "seller_id": [r[2] for r in rows],
            "price_or_rent": [1.0] * len(rows),
        }
    )


def _property_book():
    # Household 0 owns 10 (value 100) and 11 (value 100); household 1 owns 12 (value 300).
    index = pd.Index([10, 11, 12])
    return (
        pd.Series([0.0, 0.0, 1.0], index=index),
        pd.Series([100.0, 100.0, 300.0], index=index),
    )


@pytest.mark.parametrize(
    "rows,expected",
    [
        ([(10, 2, 0)], [0.5, 0.0, 0.0]),  # landlord sells one of two equal properties
        ([(10, 2, 0), (11, 3, 0)], [1.0, 0.0, 0.0]),  # sells both
        ([(12, 2, 1)], [0.0, 1.0, 0.0]),  # owner sells its only property
        ([(11, 0, 0)], [0.0, 0.0, 0.0]),  # self-sale sells nothing
    ],
)
def test_sold_collateral_share(rows, expected):
    owners, values = _property_book()
    share = sold_collateral_share(
        pre_sale_owner=owners, pre_sale_value=values, completed_sales=_sales(rows), n_households=3
    )
    np.testing.assert_allclose(share, expected)


def test_sold_collateral_share_rejects_a_seller_who_did_not_own_the_property():
    owners, values = _property_book()
    with pytest.raises(ValueError, match="must own the property"):
        sold_collateral_share(
            pre_sale_owner=owners, pre_sale_value=values, completed_sales=_sales([(12, 2, 0)]), n_households=3
        )


@pytest.mark.parametrize(
    "proceeds,expected_repayment",
    [(500.0, 90.0), (60.0, 60.0)],  # proceeds cover the debt; negative equity keeps 30 on the mortgage
)
def test_country_repayment_is_collateral_share_capped_by_proceeds(proceeds, expected_repayment):
    owners = pd.Series([0.0], index=pd.Index([10]))
    values = pd.Series([200.0], index=pd.Index([10]))
    country = Country.__new__(Country)
    country.households = SimpleNamespace(
        uses_feasibility_resolver=True,
        ts=SimpleNamespace(
            current=lambda key: {"n_households": 2, "property_sale_proceeds": np.array([proceeds, 0.0])}[key]
        ),
    )
    country.housing_market = SimpleNamespace(states={"current_sales": _sales([(10, 1, 0)])})
    country.credit_market = SimpleNamespace(mortgage_principal_after_scheduled_service=lambda: np.array([90.0, 0.0]))

    repayment = country._compute_mortgage_sale_repayment(pre_sale_owner=owners, pre_sale_value=values)

    np.testing.assert_allclose(repayment, [expected_repayment, 0.0])


def test_country_repayment_is_zero_without_the_resolver():
    country = Country.__new__(Country)
    country.households = SimpleNamespace(
        uses_feasibility_resolver=False, ts=SimpleNamespace(current=lambda key: {"n_households": 2}[key])
    )
    repayment = country._compute_mortgage_sale_repayment(
        pre_sale_owner=pd.Series(dtype=float), pre_sale_value=pd.Series(dtype=float)
    )
    np.testing.assert_array_equal(repayment, np.zeros(2))
