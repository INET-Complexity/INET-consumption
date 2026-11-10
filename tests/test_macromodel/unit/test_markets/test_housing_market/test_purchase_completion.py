"""Completion rule for matched purchases: pure fixed-point tests."""

import numpy as np
import pandas as pd
import pytest

from macromodel.markets.housing_market.purchase_completion import (
    PurchaseCompletionInputs,
    resolve_purchase_completion,
)
from macromodel.markets.housing_market.sales import move_chain_feasible


def _sales(rows: list[tuple[str, int, int, int, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["sales_types", "buyer_id", "seller_id", "property_id", "price_or_rent"])


def _inputs(
    sales: pd.DataFrame,
    *,
    n: int,
    owner: dict[int, int],
    inhabitant: dict[int, int] | None = None,
    value: dict[int, float] | None = None,
    cash_need=0.0,
    lfa=0.0,
    ifa=0.0,
    planned=0.0,
    cost=0.0,
) -> PurchaseCompletionInputs:
    properties = sorted(owner)
    inhabitant = {} if inhabitant is None else inhabitant
    value = {} if value is None else value
    return PurchaseCompletionInputs(
        current_sales=sales,
        property_owner=pd.Series([owner[p] for p in properties], index=properties),
        property_value=pd.Series([value.get(p, 100.0) for p in properties], index=properties),
        property_inhabitant=pd.Series([inhabitant.get(p, -1) for p in properties], index=properties),
        cash_need=np.broadcast_to(np.asarray(cash_need, dtype=float), (n,)).copy(),
        liquid_assets=np.broadcast_to(np.asarray(lfa, dtype=float), (n,)).copy(),
        illiquid_assets=np.broadcast_to(np.asarray(ifa, dtype=float), (n,)).copy(),
        planned_liquidation=np.broadcast_to(np.asarray(planned, dtype=float), (n,)).copy(),
        sale_cost=np.broadcast_to(np.asarray(cost, dtype=float), (n,)).copy(),
    )


def _resolve(inputs, mortgage, principal_left=None, mortgage_rejected=None):
    mortgage = np.asarray(mortgage, dtype=float)
    return resolve_purchase_completion(
        inputs,
        granted_mortgage=mortgage,
        principal_left=np.zeros_like(mortgage) if principal_left is None else np.asarray(principal_left, dtype=float),
        mortgage_rejected=mortgage_rejected,
    )


def test_purchase_completes_from_cash_left_after_the_period_cash_need():
    """LFA 50, N 20, mortgage 70: X = 30 = LFA - N, so the purchase just completes."""
    sales = _sales([("Sell", 0, 1, 10, 100.0)])
    inputs = _inputs(sales, n=2, owner={10: 1}, lfa=[50.0, 0.0], cash_need=[20.0, 0.0])

    result = _resolve(inputs, [70.0, 0.0])

    np.testing.assert_array_equal(result.kept_sales, [True])
    np.testing.assert_array_equal(result.mortgage_zeroed, [False, False])
    np.testing.assert_allclose(result.net_housing_cash_need, [30.0, -100.0])
    np.testing.assert_allclose(result.housing_lfa_draw, [30.0, 0.0])
    np.testing.assert_allclose(result.purchase_liquidation, [0.0, 0.0])
    np.testing.assert_allclose(result.sale_proceeds, [0.0, 100.0])


def test_purchase_fails_when_the_period_cash_need_takes_the_cash_and_zeroes_the_mortgage():
    """Same buyer with N = 21: one unit short, so no sale, no mortgage, no proceeds."""
    sales = _sales([("Sell", 0, 1, 10, 100.0)])
    inputs = _inputs(sales, n=2, owner={10: 1}, lfa=[50.0, 0.0], cash_need=[21.0, 0.0])

    result = _resolve(inputs, [70.0, 0.0])

    np.testing.assert_array_equal(result.kept_sales, [False])
    np.testing.assert_array_equal(result.mortgage_zeroed, [True, False])
    np.testing.assert_allclose(result.net_housing_cash_need, [0.0, 0.0])
    np.testing.assert_allclose(result.sale_proceeds, [0.0, 0.0])
    np.testing.assert_allclose(result.housing_lfa_draw, [0.0, 0.0])


def test_illiquid_sale_covers_the_rest_and_leaves_room_for_the_trade_cost():
    """X = 100, free LFA 40: sell 60 of IFA, which needs IFA >= 60 + cost."""
    sales = _sales([("Sell", 0, 1, 10, 100.0)])
    enough = _inputs(sales, n=2, owner={10: 1}, lfa=[40.0, 0.0], ifa=[65.0, 0.0], cost=[5.0, 0.0])
    short = _inputs(sales, n=2, owner={10: 1}, lfa=[40.0, 0.0], ifa=[64.0, 0.0], cost=[5.0, 0.0])

    kept = _resolve(enough, [0.0, 0.0])
    dropped = _resolve(short, [0.0, 0.0])

    np.testing.assert_array_equal(kept.kept_sales, [True])
    np.testing.assert_allclose(kept.purchase_liquidation, [60.0, 0.0])
    np.testing.assert_allclose(kept.housing_lfa_draw, [40.0, 0.0])
    np.testing.assert_array_equal(dropped.kept_sales, [False])


def test_illiquid_assets_already_planned_for_liquidation_are_not_counted_twice():
    """LFA < N with a planned forced sale of 30: only IFA - 30 - cost can pay for the house."""
    sales = _sales([("Sell", 0, 1, 10, 50.0)])
    base = dict(n=2, owner={10: 1}, lfa=[10.0, 0.0], cash_need=[40.0, 0.0], ifa=[80.0, 0.0])

    with_plan = _resolve(_inputs(sales, planned=[30.0, 0.0], cost=[1.0, 0.0], **base), [0.0, 0.0])
    without_plan = _resolve(_inputs(sales, planned=[0.0, 0.0], cost=[1.0, 0.0], **base), [0.0, 0.0])

    # 80 - 30 - 1 = 49 < 50: fails; without the plan 79 >= 50: completes.
    np.testing.assert_array_equal(with_plan.kept_sales, [False])
    np.testing.assert_array_equal(without_plan.kept_sales, [True])
    np.testing.assert_allclose(without_plan.purchase_liquidation, [50.0, 0.0])
    np.testing.assert_allclose(without_plan.housing_lfa_draw, [0.0, 0.0])


def test_mover_funds_the_purchase_with_its_own_sale_proceeds():
    """Chain: 0 buys B's home (11) for cash; B (1) buys vacant 10 with those proceeds."""
    sales = _sales([("Sell", 1, 2, 10, 100.0), ("Sell", 0, 1, 11, 100.0)])
    inputs = _inputs(sales, n=3, owner={10: 2, 11: 1}, inhabitant={11: 1}, lfa=[100.0, 0.0, 0.0])

    result = _resolve(inputs, [0.0, 0.0, 0.0])

    np.testing.assert_array_equal(result.kept_sales, [True, True])
    np.testing.assert_allclose(result.net_housing_cash_need, [100.0, 0.0, -100.0])
    np.testing.assert_allclose(result.housing_lfa_draw, [100.0, 0.0, 0.0])


def test_chain_unwinds_when_the_first_buyer_fails():
    """If 0 cannot pay, B's proceeds vanish and B's own purchase fails in the next pass."""
    sales = _sales([("Sell", 1, 2, 10, 100.0), ("Sell", 0, 1, 11, 100.0)])
    inputs = _inputs(sales, n=3, owner={10: 2, 11: 1}, inhabitant={11: 1}, lfa=[99.0, 0.0, 0.0])

    result = _resolve(inputs, [0.0, 80.0, 0.0])

    np.testing.assert_array_equal(result.kept_sales, [False, False])
    np.testing.assert_array_equal(result.mortgage_zeroed, [False, True, False])
    np.testing.assert_allclose(result.sale_proceeds, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(result.net_housing_cash_need, [0.0, 0.0, 0.0])


def test_buyer_dropped_by_the_move_chain_gets_no_mortgage():
    """1's purchase fails, so 1 stays in property 11 and 0 cannot move in, though 0 can pay."""
    sales = _sales([("Sell", 1, 2, 10, 500.0), ("Sell", 0, 1, 11, 100.0)])
    inputs = _inputs(sales, n=3, owner={10: 2, 11: 1}, inhabitant={11: 1}, lfa=[100.0, 0.0, 0.0])

    result = _resolve(inputs, [60.0, 0.0, 0.0])

    np.testing.assert_array_equal(result.kept_sales, [False, False])
    np.testing.assert_array_equal(result.mortgage_zeroed, [True, False, False])


def test_mortgage_repaid_on_sale_reduces_the_proceeds_available():
    """Proceeds 100, mortgage left 70 on the only property: only 30 funds the next home."""
    sales = _sales([("Sell", 1, 2, 10, 40.0), ("Sell", 0, 1, 11, 100.0)])
    inputs = _inputs(sales, n=3, owner={10: 2, 11: 1}, inhabitant={11: 1}, lfa=[100.0, 0.0, 0.0])

    result = _resolve(inputs, [0.0, 0.0, 0.0], principal_left=[0.0, 70.0, 0.0])

    # 1 needs 40 but nets only 30 from its sale: its purchase fails, and with
    # it the sale of its home.
    np.testing.assert_array_equal(result.kept_sales, [False, False])

    affordable = _inputs(sales, n=3, owner={10: 2, 11: 1}, inhabitant={11: 1}, lfa=[100.0, 10.0, 0.0])
    result = _resolve(affordable, [0.0, 0.0, 0.0], principal_left=[0.0, 70.0, 0.0])
    np.testing.assert_array_equal(result.kept_sales, [True, True])
    np.testing.assert_allclose(result.mortgage_sale_repayment, [0.0, 70.0, 0.0])
    np.testing.assert_allclose(result.net_housing_cash_need, [100.0, 10.0, -40.0])


def test_mortgage_test_sees_the_repayment_and_is_retested_when_the_own_sale_fails():
    """The mover's mortgage passes only while its own sale repays the old one."""
    sales = _sales([("Sell", 1, 2, 10, 100.0), ("Sell", 0, 1, 11, 100.0)])
    seen = []

    def rejected(repayment):
        seen.append(repayment.copy())
        return np.array([False, repayment[1] < 70.0, False])

    completes = _inputs(sales, n=3, owner={10: 2, 11: 1}, inhabitant={11: 1}, lfa=[100.0, 0.0, 0.0])
    result = _resolve(completes, [0.0, 70.0, 0.0], principal_left=[0.0, 70.0, 0.0], mortgage_rejected=rejected)
    np.testing.assert_array_equal(result.kept_sales, [True, True])
    np.testing.assert_array_equal(result.mortgage_zeroed, [False, False, False])
    np.testing.assert_allclose(seen[-1], [0.0, 70.0, 0.0])

    seen.clear()
    fails = _inputs(sales, n=3, owner={10: 2, 11: 1}, inhabitant={11: 1}, lfa=[99.0, 0.0, 0.0])
    result = _resolve(fails, [0.0, 70.0, 0.0], principal_left=[0.0, 70.0, 0.0], mortgage_rejected=rejected)
    np.testing.assert_array_equal(result.kept_sales, [False, False])
    np.testing.assert_array_equal(result.mortgage_zeroed, [False, True, False])
    # Last pass: no kept sale, so nothing is repaid and the old service returns in full.
    np.testing.assert_allclose(seen[-1], [0.0, 0.0, 0.0])


def test_rejected_mortgage_still_allows_a_cash_purchase():
    sales = _sales([("Sell", 0, 1, 10, 100.0)])
    inputs = _inputs(sales, n=2, owner={10: 1}, lfa=[100.0, 0.0])

    result = _resolve(inputs, [80.0, 0.0], mortgage_rejected=lambda _r: np.array([True, False]))

    np.testing.assert_array_equal(result.kept_sales, [True])
    np.testing.assert_array_equal(result.mortgage_zeroed, [True, False])
    np.testing.assert_allclose(result.housing_lfa_draw, [100.0, 0.0])


def test_landlord_repays_only_the_collateral_share_sold():
    """Two equal properties, one sold: half the mortgage is repaid from the proceeds."""
    sales = _sales([("Sell", 0, 1, 10, 100.0)])
    inputs = _inputs(sales, n=2, owner={10: 1, 11: 1}, lfa=[100.0, 0.0])

    result = _resolve(inputs, [0.0, 0.0], principal_left=[0.0, 60.0])

    np.testing.assert_allclose(result.mortgage_sale_repayment, [0.0, 30.0])
    np.testing.assert_allclose(result.net_housing_cash_need, [100.0, -70.0])


def test_self_sales_and_rentals_always_pass_and_move_no_money():
    sales = _sales([("Sell", 0, 0, 10, 100.0), ("Rental", 1, 0, 11, 5.0)])
    inputs = _inputs(sales, n=2, owner={10: 0, 11: 0})

    result = _resolve(inputs, [0.0, 0.0])

    np.testing.assert_array_equal(result.kept_sales, [True, True])
    np.testing.assert_allclose(result.net_housing_cash_need, [0.0, 0.0])
    np.testing.assert_allclose(result.sale_proceeds, [0.0, 0.0])


def test_no_sales_zeroes_any_granted_mortgage():
    inputs = _inputs(pd.DataFrame(), n=2, owner={10: 0})

    result = _resolve(inputs, [0.0, 25.0])

    assert result.kept_sales.shape == (0,)
    np.testing.assert_array_equal(result.mortgage_zeroed, [False, True])


def test_long_chain_terminates_and_drops_every_link():
    """Ten movers in a row, each needing the next one's payment; the first buyer cannot pay."""
    n = 11
    rows = [("Sell", h, h + 1, 100 + h, 100.0) for h in range(n - 1)]
    owner = {100 + h: h + 1 for h in range(n - 1)}
    inhabitant = {100 + h: h + 1 for h in range(n - 2)}  # the last seller's property is vacant
    inputs = _inputs(_sales(rows), n=n, owner=owner, inhabitant=inhabitant, lfa=0.0)

    result = _resolve(inputs, np.zeros(n))

    assert not result.kept_sales.any()

    funded = _inputs(_sales(rows), n=n, owner=owner, inhabitant=inhabitant, lfa=[100.0] + [0.0] * (n - 1))
    assert _resolve(funded, np.zeros(n)).kept_sales.all()


def test_non_finite_inputs_are_rejected():
    sales = _sales([("Sell", 0, 1, 10, 100.0)])
    inputs = _inputs(sales, n=2, owner={10: 1}, lfa=[np.nan, 0.0])

    with pytest.raises(ValueError, match="free_lfa"):
        _resolve(inputs, [0.0, 0.0])


def test_move_chain_keeps_only_sales_whose_property_is_freed():
    sales = _sales([("Sell", 0, 1, 10, 1.0), ("Sell", 2, 3, 11, 1.0), ("Sell", 1, 4, 12, 1.0)])
    inhabitants = pd.Series([1, 3, -1], index=[10, 11, 12])

    kept = move_chain_feasible(sales, np.array([True, True, True]), inhabitants)

    # 10 is freed because 1 moves to 12; 11 stays occupied by 3, who is not moving.
    np.testing.assert_array_equal(kept, [True, False, True])
