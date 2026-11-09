"""Increment (a0): sellers receive the sale price; self-sales are moves, not purchases."""

import numpy as np
import pandas as pd
import pytest

from macromodel.agents.households.func.wealth import DefaultWealthSetter
from macromodel.markets.housing_market.housing_market import HousingMarket
from macromodel.markets.housing_market.sales import self_sale_mask


def _sales(rows: list[tuple[str, int, int, int, float]]) -> pd.DataFrame:
    """Rows of (sales_type, property_id, buyer_id, seller_id, price); property value = price."""
    return pd.DataFrame(
        {
            "sales_types": [r[0] for r in rows],
            "property_id": [r[1] for r in rows],
            "property_value": [r[4] for r in rows],
            "price_or_rent": [r[4] for r in rows],
            "seller_id": [r[3] for r in rows],
            "buyer_id": [r[2] for r in rows],
        }
    )


def test_self_sale_mask_flags_only_sales_to_the_seller():
    sales = _sales(
        [
            ("Sell", 1, 0, 0, 100.0),
            ("Sell", 2, 1, 0, 100.0),
            ("Rental", 3, 2, 2, 5.0),
        ]
    )
    np.testing.assert_array_equal(self_sale_mask(sales), [True, False, False])
    assert self_sale_mask(sales.iloc[0:0]).shape == (0,)


def _two_property_owner_market(test_config) -> HousingMarket:
    return HousingMarket.from_data(
        country_name="FRA",
        scale=1,
        data=pd.DataFrame(
            {
                "House ID": [10.0, 20.0],
                "Value": [100.0, 200.0],
                "Rent": [1.0, 2.0],
                "Corresponding Owner Household ID": [0.0, 0.0],
                "Corresponding Inhabitant Household ID": [0.0, np.nan],
                "Is Owner-Occupied": [1.0, 0.0],
            }
        ),
        config=test_config["FRA"]["housing_market"],
    )


def test_self_sale_of_own_vacant_property_is_a_move_without_price(test_config):
    market = _two_property_owner_market(test_config)
    market.states["current_sales"] = _sales([("Sell", 20, 0, 0, 250.0)])
    household_states = {
        "Corresponding Inhabited House ID": np.array([10]),
        "Tenure Status of the Main Residence": np.array([1]),
        "Corresponding Property Owner": np.array([0]),
        "corr_renters": [[]],
    }

    # No mortgage and no financial wealth: a self-sale needs no financing.
    market.process_housing_market_clearing(
        household_states=household_states,
        household_received_mortgages=np.zeros(1),
        household_financial_wealth=np.zeros(1),
    )

    properties = market.states["properties"]
    assert len(market.states["current_sales"]) == 1
    assert household_states["Corresponding Inhabited House ID"][0] == 20
    assert properties.loc[20, "Corresponding Inhabitant Household ID"] == 0
    assert properties.loc[10, "Corresponding Inhabitant Household ID"] == -1
    assert properties.loc[20, "Value"] == 200.0  # not reset to the unpaid asking price
    assert market.ts.current("total_number_of_self_sales")[0] == 1
    assert market.ts.current("total_number_of_bought_houses")[0] == 0


def test_ordinary_sale_still_resets_value_and_counts_as_bought(test_config):
    market = _two_property_owner_market(test_config)
    market.states["current_sales"] = _sales([("Sell", 20, 1, 0, 250.0)])
    household_states = {
        "Corresponding Inhabited House ID": np.array([10, -1]),
        "Tenure Status of the Main Residence": np.array([1, 3]),
        "Corresponding Property Owner": np.array([0, -1]),
        "corr_renters": [[], []],
    }

    market.process_housing_market_clearing(
        household_states=household_states,
        household_received_mortgages=np.array([0.0, 250.0]),
        household_financial_wealth=np.zeros(2),
    )

    assert market.states["properties"].loc[20, "Value"] == 250.0
    assert market.ts.current("total_number_of_self_sales")[0] == 0
    assert market.ts.current("total_number_of_bought_houses")[0] == 1


def _process_household_sales(test_households, monkeypatch, sales: pd.DataFrame) -> None:
    n_households = test_households.ts.current("n_households")
    monkeypatch.setattr(
        test_households,
        "compute_rent",
        lambda **_kwargs: (np.zeros(n_households), np.zeros(n_households)),
    )
    test_households.process_housing_market_clearing(
        housing_data=pd.DataFrame(),
        social_housing_function=None,
        current_sales=sales,
        current_unemployment_benefits_by_individual=0.0,
    )


def test_completed_sales_pay_every_seller_and_skip_self_sales(test_households, monkeypatch):
    sales = _sales(
        [
            ("Sell", 1, 0, 1, 100.0),  # ordinary sale
            ("Sell", 2, 3, 2, 40.0),  # landlord 2 sells two properties
            ("Sell", 3, 4, 2, 60.0),
            ("Sell", 4, 5, 5, 999.0),  # self-sale: a move
        ]
    )

    _process_household_sales(test_households, monkeypatch, sales)

    paid = test_households.ts.current("price_paid_for_property")
    proceeds = test_households.ts.current("property_sale_proceeds")
    assert paid[0] == 100.0 and paid[3] == 40.0 and paid[4] == 60.0
    assert proceeds[1] == 100.0
    assert proceeds[2] == 100.0
    assert paid[5] == 0.0 and proceeds[5] == 0.0
    # Flow-matching residual (i): buyers' payments equal sellers' receipts.
    assert paid.sum() == pytest.approx(proceeds.sum())


def test_sale_without_household_seller_is_rejected(test_households, monkeypatch):
    with pytest.raises(ValueError, match="household seller"):
        _process_household_sales(test_households, monkeypatch, _sales([("Sell", 1, 0, -1, 100.0)]))


def test_legacy_deposit_update_adds_sale_proceeds():
    zeros = np.zeros(2)
    deposits = DefaultWealthSetter.compute_wealth_in_deposits(
        current_wealth_in_deposits=np.array([10.0, 10.0]),
        new_wealth_in_deposits=zeros,
        used_up_wealth_in_deposits=zeros,
        current_interest_paid=zeros,
        price_paid_for_property=np.array([0.0, 50.0]),
        property_sale_proceeds=np.array([50.0, 0.0]),
        debt_installments=zeros,
        new_loans=zeros,
        new_real_wealth=zeros,
        tau_cf=0.0,
    )
    np.testing.assert_allclose(deposits, [60.0, -40.0])
    # Total deposits are conserved across the two households.
    assert deposits.sum() == pytest.approx(20.0)


def test_self_sale_requests_no_mortgage(test_households, monkeypatch):
    captured = {}

    class TargetCreditStub:
        @staticmethod
        def compute_target_consumption_loans(**_kwargs):
            return np.zeros(test_households.ts.current("n_households"))

        @staticmethod
        def compute_target_mortgage(*, target_house_price, **_kwargs):
            captured["price"] = np.asarray(target_house_price, dtype=float).copy()
            return np.zeros_like(captured["price"])

    monkeypatch.setitem(test_households.functions, "target_credit", TargetCreditStub())
    test_households.uses_feasibility_resolver = False

    test_households.compute_target_credit(current_sales=_sales([("Sell", 1, 0, 0, 500.0), ("Sell", 2, 1, 2, 300.0)]))

    assert captured["price"][0] == 0.0
    assert captured["price"][1] == 300.0


def test_self_sale_of_current_home_changes_nothing(test_config):
    """Staying in one's own home must not mark it vacant (review finding on PR #173)."""
    market = _two_property_owner_market(test_config)
    vacant_before = int(market.ts.current("total_number_of_houses_unoccupied")[0])
    market.states["current_sales"] = _sales([("Sell", 10, 0, 0, 120.0)])
    household_states = {
        "Corresponding Inhabited House ID": np.array([10]),
        "Tenure Status of the Main Residence": np.array([1]),
        "Corresponding Property Owner": np.array([0]),
        "corr_renters": [[]],
    }

    market.process_housing_market_clearing(
        household_states=household_states,
        household_received_mortgages=np.zeros(1),
        household_financial_wealth=np.zeros(1),
    )

    properties = market.states["properties"]
    assert household_states["Corresponding Inhabited House ID"][0] == 10
    assert properties.loc[10, "Corresponding Inhabitant Household ID"] == 0
    assert properties.loc[10, "Corresponding Owner Household ID"] == 0
    assert properties.loc[10, "Value"] == 100.0
    assert market.ts.current("total_number_of_houses_unoccupied")[0] == vacant_before
    assert market.ts.current("total_number_of_self_sales")[0] == 1


def test_prefiltered_sales_without_type_column_drop_self_sales(test_households, monkeypatch):
    captured = {}

    class TargetCreditStub:
        @staticmethod
        def compute_target_consumption_loans(**_kwargs):
            return np.zeros(test_households.ts.current("n_households"))

        @staticmethod
        def compute_target_mortgage(*, target_house_price, **_kwargs):
            captured["price"] = np.asarray(target_house_price, dtype=float).copy()
            return np.zeros_like(captured["price"])

    monkeypatch.setitem(test_households.functions, "target_credit", TargetCreditStub())
    test_households.uses_feasibility_resolver = False
    prefiltered = _sales([("Sell", 1, 0, 0, 500.0), ("Sell", 2, 1, 2, 300.0)]).drop(columns="sales_types")

    test_households.compute_target_credit(current_sales=prefiltered)

    assert captured["price"][0] == 0.0
    assert captured["price"][1] == 300.0


@pytest.mark.parametrize("proceeds", [0.0, 30.0])
def test_post_labour_refresh_counts_sale_proceeds_as_cash(test_households, proceeds):
    """Support base before goods settlement: proceeds cut the residual need one for one."""
    from macromodel.agents.households.func.financial_feasibility import PostGrantFeasiblePlan

    hh = test_households
    n = hh.ts.current("n_households")
    zeros = np.zeros(n)
    hh.post_grant_feasible_plan = PostGrantFeasiblePlan(
        credit_granted=zeros.copy(),
        credit_rationing_gap=zeros.copy(),
        planned_liquidation_total=zeros.copy(),
        reserved_liquidation_total=zeros.copy(),
        residual_shortfall_after_granted_credit=zeros.copy(),
    )
    hh._known_receipts_ready = False
    hh.ts.override_current("expected_income", zeros.copy())
    hh.ts.override_current("target_consumption", np.zeros_like(hh.ts.current("target_consumption")))
    hh.ts.override_current("target_investment", np.zeros_like(hh.ts.current("target_investment")))
    hh.ts.override_current("rent", zeros.copy())
    hh.ts.override_current("liquid_financial_assets", zeros.copy())
    hh.ts.override_current("price_paid_for_property", zeros.copy())
    hh.ts.override_current("received_mortgages", zeros.copy())
    hh.ts.override_current("property_sale_proceeds", np.full(n, proceeds))

    hh.refresh_post_labour_feasibility(scheduled_debt_service=np.full(n, 100.0))

    np.testing.assert_allclose(hh.post_grant_feasible_plan.residual_shortfall_after_lfa, np.full(n, 100.0 - proceeds))
    np.testing.assert_allclose(
        hh.post_grant_feasible_plan.residual_shortfall_after_granted_credit, np.full(n, 100.0 - proceeds)
    )
