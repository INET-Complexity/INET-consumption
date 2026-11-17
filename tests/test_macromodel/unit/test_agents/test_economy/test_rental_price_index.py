"""Rent price indices for the three housing flows in GDP (issue #197)."""

import numpy as np
import pandas as pd
import pytest

from macromodel.country.country import floor_initial_rents
from macromodel.economy.economy import Economy
from macromodel.timeseries import TimeSeries

TENANT, OWNER, VACANT = 3, 1, -1


def make_economy(previous_social_rent=np.nan):
    economy = Economy.__new__(Economy)
    economy.ts = TimeSeries(
        rpi_actual=[1.0],
        rpi_actual_inflation=[0.0],
        rpi_imputed=[1.0],
        rpi_imputed_inflation=[0.0],
        social_rent_price=[1.0],
        social_rent_price_inflation=[0.0],
        social_rent_per_adult=[previous_social_rent],
    )
    return economy


def compute(economy, current, previous, tenure, social=1.0):
    economy.compute_rent_price_indices(
        current_property_rents=np.array(current, dtype=float),
        previous_property_rents=np.array(previous, dtype=float),
        dwelling_tenure=np.array(tenure),
        social_rent_per_adult=social,
    )


def test_actual_and_imputed_rents_are_indexed_separately():
    economy = make_economy()
    compute(economy, [11.0, 21.0, 30.0], [10.0, 20.0, 30.0], [TENANT, TENANT, OWNER])
    assert economy.ts.current("rpi_actual")[0] == pytest.approx(32.0 / 30.0)
    assert economy.ts.current("rpi_imputed")[0] == pytest.approx(1.0)


def test_relet_dwelling_is_matched_on_its_previous_asking_rent():
    # Dwelling 0 was vacant (asking rent 10) and is let at t at the floored rent 15.
    economy = make_economy()
    compute(economy, [15.0, 20.0], [10.0, 20.0], [TENANT, TENANT])
    assert economy.ts.current("rpi_actual")[0] == pytest.approx(35.0 / 30.0)


def test_vacant_dwellings_and_dwellings_without_previous_rent_are_excluded():
    economy = make_economy()
    compute(economy, [99.0, 12.0, 50.0], [10.0, 10.0, 0.0], [VACANT, TENANT, TENANT])
    assert economy.ts.current("rpi_actual")[0] == pytest.approx(1.2)


def test_empty_group_holds_the_index_flat():
    economy = make_economy()
    compute(economy, [12.0], [10.0], [TENANT])
    assert economy.ts.current("rpi_imputed")[0] == 1.0


def test_social_price_follows_rent_per_adult_and_is_flat_on_first_period():
    economy = make_economy()
    compute(economy, [10.0], [10.0], [TENANT], social=2.0)
    assert economy.ts.current("social_rent_price")[0] == 1.0
    compute(economy, [10.0], [10.0], [TENANT], social=2.2)
    assert economy.ts.current("social_rent_price")[0] == pytest.approx(1.1)


def test_mismatched_dwelling_sets_are_rejected():
    with pytest.raises(ValueError, match="same dwellings"):
        compute(make_economy(), [1.0, 2.0], [1.0], [TENANT, TENANT])


def test_initial_rents_are_floored_and_passed_to_household_flows():
    housing = pd.DataFrame({"Value": [1000.0, 1000.0, 1000.0], "Rent": [2.0, 9.0, 1.0]}, index=[0, 1, 2])
    households = pd.DataFrame(
        {
            "Corresponding Inhabited House ID": [0, 1, 2, -1],
            "Tenure Status of the Main Residence": [TENANT, TENANT, OWNER, -1],
            "Rent Paid": [2.0, 9.0, 0.0, 4.0],
            "Rent Imputed": [0.0, 0.0, 1.0, 0.0],
        }
    )
    # 2% a year on 1000 is 5 a quarter.
    floor_initial_rents(households, housing, minimum_annual_rental_yield=0.02, time_unit=3)
    np.testing.assert_allclose(housing["Rent"], [5.0, 9.0, 5.0])
    np.testing.assert_allclose(households["Rent Paid"], [5.0, 9.0, 0.0, 4.0])
    np.testing.assert_allclose(households["Rent Imputed"], [0.0, 0.0, 5.0, 0.0])


def test_zero_floor_leaves_initial_rents_unchanged():
    housing = pd.DataFrame({"Value": [1000.0], "Rent": [1.0]})
    households = pd.DataFrame(
        {
            "Corresponding Inhabited House ID": [0],
            "Tenure Status of the Main Residence": [TENANT],
            "Rent Paid": [1.0],
            "Rent Imputed": [0.0],
        }
    )
    floor_initial_rents(households, housing, minimum_annual_rental_yield=0.0, time_unit=3)
    assert housing["Rent"].tolist() == [1.0]
