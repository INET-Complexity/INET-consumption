from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from macromodel.configurations import HousingMarketConfiguration
from macromodel.markets.housing_market.func.clearing import (
    AutomaticHousingMarketClearer,
    DefaultHousingMarketClearer,
)
from macromodel.markets.housing_market.housing_market import HousingMarket


def _mixed_property_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "House ID": [10.0, 11.0, 12.0],
            "Value": [100.0, 200.0, 300.0],
            "Rent": [1.0, 2.0, 3.0],
            "Corresponding Owner Household ID": [0.0, 1.0, 2.0],
            "Corresponding Inhabitant Household ID": [0.0, 3.0, np.nan],
            "Is Owner-Occupied": [1.0, 0.0, 0.0],
        }
    )


def _moving_owner_property_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "House ID": [10.0, 20.0, 30.0],
            "Value": [100.0, 200.0, 0.0],
            "Rent": [1.0, 2.0, 0.0],
            "Corresponding Owner Household ID": [0.0, 1.0, np.nan],
            "Corresponding Inhabitant Household ID": [0.0, np.nan, np.nan],
            "Is Owner-Occupied": [1.0, 0.0, 1.0],
        }
    )


def _owner_occupied_flags(properties: pd.DataFrame) -> np.ndarray:
    return (
        (properties["Corresponding Owner Household ID"] >= 0)
        & (properties["Corresponding Owner Household ID"] == properties["Corresponding Inhabitant Household ID"])
    ).astype(int)


def _assert_mixed_property_state(market: HousingMarket) -> None:
    properties = market.states["properties"]
    np.testing.assert_array_equal(properties.index.values, np.array([10, 11, 12]))
    np.testing.assert_array_equal(properties["House ID"].values, np.array([10, 11, 12]))
    np.testing.assert_array_equal(properties["Corresponding Owner Household ID"].values, np.array([0, 1, 2]))
    np.testing.assert_array_equal(properties["Corresponding Inhabitant Household ID"].values, np.array([0, 3, -1]))
    np.testing.assert_array_equal(properties["Is Owner-Occupied"].values, np.array([1, 0, 0]))
    assert market.ts.current("total_number_of_houses_owner_occupied")[0] == 1
    assert market.ts.current("total_number_of_houses_rented")[0] == 1
    assert market.ts.current("total_number_of_houses_unoccupied")[0] == 1


def _process_owner_move(market: HousingMarket, sales_type: str) -> None:
    price_or_rent = 250.0 if sales_type == "Sell" else 2.0
    market.states["current_sales"] = pd.DataFrame(
        {
            "sales_types": [sales_type],
            "property_id": [20],
            "property_value": [200.0],
            "price_or_rent": [price_or_rent],
            "seller_id": [1],
            "buyer_id": [0],
        }
    )
    household_states = {
        "Corresponding Inhabited House ID": np.array([10, 20]),
        "Tenure Status of the Main Residence": np.array([1, 1]),
        "Corresponding Property Owner": np.array([0, 1]),
        "corr_renters": [[], []],
    }
    received_mortgages = np.array([price_or_rent, 0.0])
    financial_wealth = np.array([0.0, 0.0])

    market.process_housing_market_clearing(
        household_states=household_states,
        household_received_mortgages=received_mortgages,
        household_financial_wealth=financial_wealth,
    )


def test_households_prepare_housing_market_marks_missing_inhabitants_for_rent(test_households, test_config):
    market = HousingMarket.from_data(
        country_name="FRA",
        scale=1,
        data=_mixed_property_data(),
        config=test_config["FRA"]["housing_market"],
    )
    housing_data = market.states["properties"]

    test_households.prepare_housing_market_clearing(
        housing_data=housing_data,
        observed_fraction_value_price=np.array([1.0, 0.0]),
        observed_fraction_rent_value=np.array([0.01, 0.0]),
        expected_hpi_growth=0.0,
        assumed_mortgage_maturity=120,
        rental_income_taxes=0.0,
    )

    assert bool(housing_data.loc[12, "Up for Rent"])
    assert bool(housing_data.loc[12, "Newly on the Rental Market"])


def test_households_prepare_housing_market_marks_first_row_vacancy_as_new(test_households, test_config):
    housing_data = _mixed_property_data()
    housing_data["House ID"] = [0.0, 1.0, 2.0]
    housing_data["Corresponding Inhabitant Household ID"] = [-1, 3, 4]
    housing_data["Newly on the Rental Market"] = False
    housing_data["Up for Rent"] = False
    housing_data["Temporarily for Sale"] = False
    housing_data["Sale Price"] = np.nan
    market = HousingMarket.from_data(
        country_name="FRA",
        scale=1,
        data=housing_data,
        config=test_config["FRA"]["housing_market"],
    )
    housing_data = market.states["properties"]

    test_households.prepare_housing_market_clearing(
        housing_data=housing_data,
        observed_fraction_value_price=np.array([1.0, 0.0]),
        observed_fraction_rent_value=np.array([0.01, 0.0]),
        expected_hpi_growth=0.0,
        assumed_mortgage_maturity=120,
        rental_income_taxes=0.0,
    )

    assert bool(housing_data.loc[0, "Up for Rent"])
    assert bool(housing_data.loc[0, "Newly on the Rental Market"])


def test_main_residence_wealth_uses_house_id_index(test_households, test_config):
    market = HousingMarket.from_data(
        country_name="FRA",
        scale=1,
        data=_mixed_property_data(),
        config=test_config["FRA"]["housing_market"],
    )
    test_households.states["Tenure Status of the Main Residence"][:] = 0
    test_households.states["Corresponding Inhabited House ID"][:] = -1
    test_households.states["Tenure Status of the Main Residence"][0] = 1
    test_households.states["Corresponding Inhabited House ID"][0] = 10

    wealth_main_residence = test_households.compute_wealth_of_the_main_residence(
        housing_data=market.states["properties"]
    )

    assert wealth_main_residence[0] == 100.0
    assert np.count_nonzero(wealth_main_residence) == 1


def test_owned_vacant_previous_residence_counts_as_other_property_wealth(test_households, test_config):
    market = HousingMarket.from_data(
        country_name="FRA",
        scale=1,
        data=_moving_owner_property_data(),
        config=test_config["FRA"]["housing_market"],
    )
    _process_owner_move(market, "Sell")

    test_households.states["Tenure Status of the Main Residence"][:] = 0
    test_households.states["Corresponding Inhabited House ID"][:] = -1
    test_households.states["Tenure Status of the Main Residence"][0] = 1
    test_households.states["Corresponding Inhabited House ID"][0] = 20

    wealth_main_residence = test_households.compute_wealth_of_the_main_residence(
        housing_data=market.states["properties"]
    )
    wealth_other_properties = test_households.compute_wealth_of_other_properties(
        housing_data=market.states["properties"]
    )

    assert wealth_main_residence[0] == 250.0
    assert wealth_other_properties[0] == 100.0
    assert wealth_other_properties.sum() == 100.0


@pytest.mark.parametrize(
    "clearer",
    [
        DefaultHousingMarketClearer(random_assignment_shock_variance=0.0),
        AutomaticHousingMarketClearer(random_assignment_shock_variance=0.0),
    ],
)
@pytest.mark.parametrize(
    ("is_rental_market", "status_field", "price_field", "property_id", "expected_sales_type"),
    [
        (False, "Temporarily for Sale", "Sale Price", 11, "Sell"),
        (True, "Up for Rent", "Rent", 12, "Rental"),
    ],
)
def test_clearers_use_house_id_labels_for_open_properties(
    clearer,
    is_rental_market,
    status_field,
    price_field,
    property_id,
    expected_sales_type,
    test_config,
):
    market = HousingMarket.from_data(
        country_name="FRA",
        scale=1,
        data=_mixed_property_data(),
        config=test_config["FRA"]["housing_market"],
    )
    housing_data = market.states["properties"]
    housing_data["Temporarily for Sale"] = False
    housing_data["Up for Rent"] = False
    housing_data.loc[property_id, status_field] = True

    max_willing_to_pay = np.zeros(5)
    max_willing_to_pay[4] = housing_data.loc[property_id, price_field] + 1.0

    transactions = clearer.perform_matching(
        housing_data=housing_data,
        household_main_residence_tenure_status=np.zeros(5),
        max_willing_to_pay=max_willing_to_pay,
        is_rental_market=is_rental_market,
    )

    assert transactions.loc[0, "sales_types"] == expected_sales_type
    assert transactions.loc[0, "property_id"] == property_id
    assert transactions.loc[0, "buyer_id"] == 4
    assert not bool(housing_data.loc[property_id, status_field])


@pytest.mark.parametrize(
    ("sales_type", "expected_new_property_flag", "expected_owner_occupied_count", "expected_rented_count"),
    [
        ("Sell", 1, 1, 0),
        ("Rental", 0, 0, 1),
    ],
)
def test_process_clearing_refreshes_owner_occupied_flags_after_owner_move(
    sales_type,
    expected_new_property_flag,
    expected_owner_occupied_count,
    expected_rented_count,
    test_config,
):
    market = HousingMarket.from_data(
        country_name="FRA",
        scale=1,
        data=_moving_owner_property_data(),
        config=test_config["FRA"]["housing_market"],
    )

    _process_owner_move(market, sales_type)
    properties = market.states["properties"]

    assert properties.loc[10, "Corresponding Inhabitant Household ID"] == -1
    assert properties.loc[10, "Is Owner-Occupied"] == 0
    assert properties.loc[20, "Is Owner-Occupied"] == expected_new_property_flag
    np.testing.assert_array_equal(properties["Is Owner-Occupied"].values, _owner_occupied_flags(properties).values)
    assert market.ts.current("total_number_of_houses_owner_occupied")[0] == expected_owner_occupied_count
    assert market.ts.current("total_number_of_houses_rented")[0] == expected_rented_count
    assert market.ts.current("total_number_of_houses_unoccupied")[0] == 2


def test_from_data_preserves_property_identifiers_and_initial_counts(test_config):
    market = HousingMarket.from_data(
        country_name="FRA",
        scale=1,
        data=_mixed_property_data(),
        config=test_config["FRA"]["housing_market"],
    )

    _assert_mixed_property_state(market)


def test_from_data_uses_guarded_owner_occupied_rule(test_config):
    market = HousingMarket.from_data(
        country_name="FRA",
        scale=1,
        data=_moving_owner_property_data(),
        config=test_config["FRA"]["housing_market"],
    )

    properties = market.states["properties"]

    assert properties.loc[30, "Corresponding Owner Household ID"] == -1
    assert properties.loc[30, "Corresponding Inhabitant Household ID"] == -1
    assert properties.loc[30, "Is Owner-Occupied"] == 0
    np.testing.assert_array_equal(properties["Is Owner-Occupied"].values, _owner_occupied_flags(properties).values)


def test_from_pickled_market_preserves_property_identifiers_and_initial_counts():
    market = HousingMarket.from_pickled_market(
        synthetic_housing_market=SimpleNamespace(housing_market_data=_mixed_property_data()),
        housing_market_configuration=HousingMarketConfiguration(),
        scale=1,
        country_name="FRA",
    )

    _assert_mixed_property_state(market)
