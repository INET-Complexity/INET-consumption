"""Increment (b): the purchase completion rule inside credit clearing and at the reservation."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from macromodel.agents.households.func.financial_feasibility import (
    HouseholdFinancialFeasibility,
    PostGrantFeasiblePlan,
    forced_liquidation_cash,
)
from macromodel.markets.credit_market.credit_market import CreditMarket
from macromodel.markets.housing_market.housing_market import HousingMarket
from macromodel.markets.housing_market.purchase_completion import PurchaseCompletionInputs

N_HOUSEHOLDS = 3
MATURITY = 12


def _market() -> CreditMarket:
    """Two banks, three households; household 0 owes 100 at 1% with an instalment of 10."""
    cons = np.zeros((3, 2, N_HOUSEHOLDS))
    mortgage = np.zeros_like(cons)
    mortgage[:, 0, 0] = [100, 0.01, 10]
    return CreditMarket.from_data("TST", np.zeros((3, 2, 1)), np.zeros((3, 2, 1)), cons, mortgage)


def _households() -> SimpleNamespace:
    return SimpleNamespace(
        consumer_underwriting_income=np.full(N_HOUSEHOLDS, 100.0),
        consumer_underwriting_other_service=np.array([2.0, 0.0, 0.0]),
        consumer_underwriting_dsti_limit=0.35,
        uses_feasibility_resolver=True,
        current_ficp_active=lambda: np.zeros(N_HOUSEHOLDS, dtype=bool),
    )


def _sales(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["sales_types", "buyer_id", "seller_id", "property_id", "price_or_rent"])


def _mover_inputs(*, home_price: float, lfa) -> PurchaseCompletionInputs:
    """Household 0 buys vacant property 10 from household 2 for 150 and sells its home 11 to household 1."""
    return PurchaseCompletionInputs(
        current_sales=_sales([("Sell", 0, 2, 10, 150.0), ("Sell", 1, 0, 11, home_price)]),
        property_owner=pd.Series([2, 0], index=[10, 11]),
        property_value=pd.Series([150.0, 120.0], index=[10, 11]),
        property_inhabitant=pd.Series([-1, 0], index=[10, 11]),
        cash_need=np.zeros(N_HOUSEHOLDS),
        liquid_assets=np.asarray(lfa, dtype=float),
        illiquid_assets=np.zeros(N_HOUSEHOLDS),
        planned_liquidation=np.zeros(N_HOUSEHOLDS),
        sale_cost=np.zeros(N_HOUSEHOLDS),
    )


def _new_mortgage(instalment: float, household: int = 0) -> np.ndarray:
    mortgage = np.zeros((3, 2, N_HOUSEHOLDS))
    mortgage[:, 1, household] = [100.0, 0.01, instalment]
    return mortgage


def _complete(market, mortgage, inputs):
    return market._complete_property_purchases(
        _households(), MATURITY, np.zeros((3, 2, N_HOUSEHOLDS)), mortgage, inputs
    )


def test_mover_whose_sale_repays_the_old_mortgage_is_tested_on_the_new_one_only():
    """Old service 10 + new 26 + other 2 = 38 > 35 fails today; the sale repays B = 91, leaving 28."""
    market = _market()
    mortgage = _new_mortgage(26.0)

    _, legacy = market.enforce_household_origination_dsti(_households(), MATURITY, np.zeros_like(mortgage), mortgage)
    assert legacy[0].sum() == 0.0  # characterisation: rejected on the full old service

    _, kept = _complete(market, mortgage, _mover_inputs(home_price=120.0, lfa=[21.0, 120.0, 0.0]))

    np.testing.assert_array_equal(kept, mortgage)
    completion = market.current_purchase_completion()
    np.testing.assert_array_equal(completion.kept_sales, [True, True])
    np.testing.assert_allclose(completion.mortgage_sale_repayment, [91.0, 0.0, 0.0])
    # X = 150 - 100 - (120 - 91) = 21, all from liquid assets.
    np.testing.assert_allclose(completion.net_housing_cash_need, [21.0, 120.0, -150.0])
    np.testing.assert_allclose(completion.housing_lfa_draw, [21.0, 120.0, 0.0])


def test_mover_with_negative_equity_keeps_the_service_on_the_unrepaid_debt():
    """Instalment 29: passes if the sale repays all 91, fails if it repays only 50 (10 * 41/91 stays)."""
    full = _market()
    _, kept = _complete(full, _new_mortgage(29.0), _mover_inputs(home_price=120.0, lfa=[21.0, 120.0, 0.0]))
    assert kept[0].sum() == 100.0

    short = _market()
    _, rejected = _complete(short, _new_mortgage(29.0), _mover_inputs(home_price=50.0, lfa=[500.0, 50.0, 0.0]))
    assert rejected[0].sum() == 0.0
    completion = short.current_purchase_completion()
    np.testing.assert_allclose(completion.mortgage_sale_repayment, [50.0, 0.0, 0.0])
    # The mover still buys from cash: 150 - 0 - (50 - 50) = 150.
    np.testing.assert_array_equal(completion.kept_sales, [True, True])
    np.testing.assert_allclose(completion.net_housing_cash_need[0], 150.0)


def test_mover_whose_own_sale_fails_is_retested_on_the_full_old_service():
    """The home buyer cannot pay, so nothing is repaid: the mortgage fails and the purchase with it."""
    market = _market()
    before = market.states["mort_loans"].copy()

    _, mort = _complete(market, _new_mortgage(26.0), _mover_inputs(home_price=120.0, lfa=[21.0, 119.0, 0.0]))

    assert mort[0].sum() == 0.0
    completion = market.current_purchase_completion()
    np.testing.assert_array_equal(completion.kept_sales, [False, False])
    np.testing.assert_array_equal(completion.mortgage_zeroed, [True, False, False])
    np.testing.assert_allclose(completion.sale_proceeds, 0.0)
    np.testing.assert_allclose(completion.purchase_liquidation, 0.0)

    # Booking the cleared mortgages leaves the buyer's existing loan untouched.
    market._add_new_loans("mort_loans", mort)
    np.testing.assert_array_equal(market.states["mort_loans"], before)


def test_buyer_without_a_sale_is_tested_exactly_as_before():
    """Characterisation: with no own sale the rejection set equals the pre-(b) test."""
    inputs = PurchaseCompletionInputs(
        current_sales=_sales([("Sell", 0, 2, 10, 150.0), ("Sell", 1, 2, 12, 150.0)]),
        property_owner=pd.Series([2, 0, 2], index=[10, 11, 12]),
        property_value=pd.Series([150.0, 120.0, 150.0], index=[10, 11, 12]),
        property_inhabitant=pd.Series([-1, 0, -1], index=[10, 11, 12]),
        cash_need=np.zeros(N_HOUSEHOLDS),
        liquid_assets=np.full(N_HOUSEHOLDS, 1000.0),
        illiquid_assets=np.zeros(N_HOUSEHOLDS),
        planned_liquidation=np.zeros(N_HOUSEHOLDS),
        sale_cost=np.zeros(N_HOUSEHOLDS),
    )
    for instalment_0, instalment_1 in [(20.0, 30.0), (26.0, 36.0), (23.0, 35.0)]:
        mortgage = _new_mortgage(instalment_0) + _new_mortgage(instalment_1, household=1)
        consumer = np.zeros_like(mortgage)
        consumer[:, 0, 1] = [50.0, 0.02, 5.0]
        legacy_cons, legacy_mort = _market().enforce_household_origination_dsti(
            _households(), MATURITY, consumer, mortgage
        )
        market = _market()
        cons, mort = market._complete_property_purchases(_households(), MATURITY, consumer, mortgage, inputs)
        np.testing.assert_array_equal(cons, legacy_cons)
        np.testing.assert_array_equal(mort, legacy_mort)


def test_without_underwriting_snapshot_only_the_completion_test_applies():
    market = _market()
    households = SimpleNamespace(uses_feasibility_resolver=True)
    mortgage = _new_mortgage(90.0)  # would fail any DSTI test

    _, mort = market._complete_property_purchases(
        households,
        MATURITY,
        np.zeros_like(mortgage),
        mortgage,
        _mover_inputs(home_price=120.0, lfa=[21.0, 120.0, 0.0]),
    )

    np.testing.assert_array_equal(mort, mortgage)


def test_principal_left_matches_the_service_snapshot_used_for_the_repayment():
    """The completion rule and the booked repayment must read the same principal left."""
    market = _market()
    _complete(market, _new_mortgage(26.0), _mover_inputs(home_price=120.0, lfa=[21.0, 120.0, 0.0]))
    market.prepare_household_service_snapshot()

    np.testing.assert_allclose(market.mortgage_principal_after_scheduled_service(), [91.0, 0.0, 0.0])
    np.testing.assert_allclose(market.current_purchase_completion().mortgage_sale_repayment, [91.0, 0.0, 0.0])


# --- Reservation -----------------------------------------------------------------


def _plan(*, planned, residual_after_lfa) -> PostGrantFeasiblePlan:
    n = len(planned)
    return PostGrantFeasiblePlan(
        credit_granted=np.zeros(n),
        credit_rationing_gap=np.zeros(n),
        planned_liquidation_total=np.asarray(planned, dtype=float),
        residual_shortfall_after_granted_credit=np.zeros(n),
        residual_shortfall_after_lfa=np.asarray(residual_after_lfa, dtype=float),
    )


@pytest.mark.parametrize(
    "planned,purchase,gross,forced_cash",
    [
        (0.0, 60.0, 65.0, 0.0),  # purchase sale only: gross adds the cost once
        (30.0, 60.0, 95.0, 30.0),  # forced and purchase sale share one cost
        (30.0, 0.0, 35.0, 30.0),  # no purchase: as in increment (a)
    ],
)
def test_reservation_adds_the_purchase_sale_and_charges_the_cost_once(planned, purchase, gross, forced_cash):
    feasibility = HouseholdFinancialFeasibility()
    plan = feasibility.reserve_executable_liquidation(
        _plan(planned=[planned], residual_after_lfa=[planned]),
        available_pre_stage4_ifa=np.array([100.0]),
        sale_cost=np.array([5.0]),
        purchase_liquidation=np.array([purchase]),
    )

    assert plan.planned_liquidation_total[0] == planned  # forced need only
    assert plan.purchase_liquidation_total[0] == purchase
    assert plan.reserved_liquidation_total[0] == pytest.approx(gross)
    assert plan.liquidation_sale_cost[0] == pytest.approx(5.0)
    assert forced_liquidation_cash(plan)[0] == pytest.approx(forced_cash)
    # The purchase sale does not reduce the cash shortfall.
    assert plan.residual_shortfall_after_granted_credit[0] == pytest.approx(0.0)

    refreshed = feasibility.refresh_cash_needs(
        plan, cash_uses=np.array([100.0]), cash_income=np.array([10.0]), available_lfa=np.array([0.0])
    )
    assert refreshed.residual_shortfall_after_granted_credit[0] == pytest.approx(90.0 - forced_cash)


def test_reservation_rejects_a_purchase_sale_the_illiquid_assets_cannot_fund():
    with pytest.raises(RuntimeError, match="completed property purchase"):
        HouseholdFinancialFeasibility().reserve_executable_liquidation(
            _plan(planned=[30.0], residual_after_lfa=[30.0]),
            available_pre_stage4_ifa=np.array([90.0]),
            sale_cost=np.array([5.0]),
            purchase_liquidation=np.array([60.0]),
        )


# --- Housing market --------------------------------------------------------------


def _market_with_two_sales(test_config) -> tuple[HousingMarket, dict]:
    market = HousingMarket.from_data(
        country_name="FRA",
        scale=1,
        data=pd.DataFrame(
            {
                "House ID": [10.0, 20.0],
                "Value": [100.0, 200.0],
                "Rent": [1.0, 2.0],
                "Corresponding Owner Household ID": [0.0, 0.0],
                "Corresponding Inhabitant Household ID": [np.nan, np.nan],
                "Is Owner-Occupied": [0.0, 0.0],
            }
        ),
        config=test_config["FRA"]["housing_market"],
    )
    market.states["current_sales"] = _sales([("Sell", 1, 0, 10, 110.0), ("Sell", 2, 0, 20, 210.0)])
    household_states = {
        "Corresponding Inhabited House ID": np.array([-1, -1, -1]),
        "Tenure Status of the Main Residence": np.array([3, 3, 3]),
        "Corresponding Property Owner": np.array([0, -1, -1]),
        "corr_renters": [[], [], []],
    }
    return market, household_states


def test_completed_sale_mask_is_authoritative_over_mortgages_and_wealth(test_config):
    """Buyer 1 holds a mortgage and wealth but its purchase did not complete; buyer 2 has neither."""
    market, household_states = _market_with_two_sales(test_config)

    market.process_housing_market_clearing(
        household_states=household_states,
        household_received_mortgages=np.array([0.0, 110.0, 0.0]),
        household_financial_wealth=np.array([0.0, 500.0, 0.0]),
        kept_sales=np.array([False, True]),
    )

    properties = market.states["properties"]
    assert len(market.states["current_sales"]) == 1
    assert properties.loc[10, "Corresponding Owner Household ID"] == 0
    assert properties.loc[10, "Value"] == 100.0
    assert properties.loc[20, "Corresponding Owner Household ID"] == 2
    assert properties.loc[20, "Value"] == 210.0
    assert market.ts.current("total_number_of_bought_houses")[0] == 1


def test_completed_sale_mask_must_cover_every_matched_row(test_config):
    market, household_states = _market_with_two_sales(test_config)

    with pytest.raises(ValueError, match="one entry per matched sale"):
        market.process_housing_market_clearing(
            household_states=household_states,
            household_received_mortgages=np.zeros(3),
            household_financial_wealth=np.zeros(3),
            kept_sales=np.array([True]),
        )
