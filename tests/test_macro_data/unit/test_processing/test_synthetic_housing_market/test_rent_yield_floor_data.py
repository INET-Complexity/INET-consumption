"""Rent/value floor applied to the initial rents in the data step (issue #197)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from macro_data.configuration.dataconfiguration import CountryDataConfiguration
from macro_data.processing.synthetic_matching.matching_households_with_houses import apply_rent_yield_floor

TENANT, OWNER = 3, 1


def make_data(rents, house_ids=(10, 11, 12)):
    # The first two dwellings are let by household 3 to households 0 and 1; the third is owner-occupied by 2.
    housing = pd.DataFrame(
        {
            "House ID": list(house_ids),
            "Value": [1000.0, 1000.0, 1000.0],
            "Rent": rents,
            "Corresponding Owner Household ID": [3, 3, 2],
            "Corresponding Inhabitant Household ID": [0.0, 1.0, 2.0],
            "Is Owner-Occupied": [False, False, True],
        }
    )
    households = pd.DataFrame(
        {
            "Corresponding Inhabited House ID": [house_ids[0], house_ids[1], house_ids[2], np.nan],
            "Tenure Status of the Main Residence": [TENANT, TENANT, OWNER, OWNER],
            "Rent Paid": [rents[0], rents[1], 0.0, 0.0],
            "Rent Imputed": [0.0, 0.0, rents[2], 0.0],
            "Rental Income from Real Estate": [0.0, 0.0, 0.0, 0.75 * (rents[0] + rents[1])],
        }
    )
    return housing, households


def floor(households, housing, phi=0.02):
    # 2% a year on 1000 is 5 a quarter; rental income tax 25%.
    apply_rent_yield_floor(households, housing, phi, time_unit=3, rental_income_taxes=0.25)


def test_floor_raises_dwelling_rents_and_both_sides_of_each_flow():
    housing, households = make_data([2.0, 9.0, 1.0])
    floor(households, housing)
    np.testing.assert_allclose(housing["Rent"], [5.0, 9.0, 5.0])
    np.testing.assert_allclose(households["Rent Paid"], [5.0, 9.0, 0.0, 0.0])
    np.testing.assert_allclose(households["Rent Imputed"], [0.0, 0.0, 5.0, 0.0])
    # Landlord income stays equal to the rents of its let dwellings, net of tax.
    np.testing.assert_allclose(households["Rental Income from Real Estate"], [0.0, 0.0, 0.0, 0.75 * 14.0])


def test_households_are_matched_on_house_id_not_row_position():
    housing, households = make_data([2.0, 9.0, 1.0], house_ids=(12, 10, 11))
    floor(households, housing)
    np.testing.assert_allclose(households["Rent Paid"], [5.0, 9.0, 0.0, 0.0])
    np.testing.assert_allclose(households["Rent Imputed"], [0.0, 0.0, 5.0, 0.0])


def test_zero_floor_changes_nothing():
    housing, households = make_data([1.0, 1.0, 1.0])
    floor(households, housing, phi=0.0)
    assert housing["Rent"].tolist() == [1.0, 1.0, 1.0]
    assert households["Rental Income from Real Estate"].tolist()[-1] == pytest.approx(1.5)


def fra_country_data_config() -> dict:
    root = Path(__file__).resolve().parents[5]
    with (root / "run_model/config/data_config_FRA.yaml").open() as f:
        return yaml.safe_load(f)["country_configs"]["FRA"]


def test_fra_data_configuration_sets_the_floor():
    assert CountryDataConfiguration(**fra_country_data_config()).minimum_annual_rental_yield == 0.02


def test_negative_floor_is_rejected():
    config = fra_country_data_config()
    config["minimum_annual_rental_yield"] = -0.01
    with pytest.raises(ValueError, match="minimum_annual_rental_yield"):
        CountryDataConfiguration(**config)


def test_floor_defaults_to_zero():
    config = fra_country_data_config()
    del config["minimum_annual_rental_yield"]
    assert CountryDataConfiguration(**config).minimum_annual_rental_yield == 0.0
