"""Atomic accounting, initial anchor and first comparable growth contracts."""

from copy import deepcopy

import h5py
import numpy as np
import pytest

from macromodel.agents.households.households_ts import align_initial_goods_consumption
from macromodel.agents.individuals.individual_properties import ActivityStatus
from macromodel.economy.economy_ts import create_economy_timeseries
from tests.test_macromodel.unit.test_agents.test_economy.test_total_consumption_increment0 import CASES

BRIDGES = (
    "owner_occupied_housing_output",
    "owner_occupied_housing_value_added",
    "owner_occupied_housing_operating_income",
)


def initial_accounts(housing, cash_rent=100.0):
    """Independent balanced input accounts: old GDP=450, new GDP=450+H."""
    sales = np.full(18, 371.0 / 18)
    return create_economy_timeseries(
        country_name="FRA",
        all_country_names=["FRA"],
        n_industries=18,
        initial_firm_prices=np.ones(18),
        initial_firm_total_sales=371.0,
        initial_sectoral_firm_sales=sales,
        initial_sectoral_firm_prices=np.ones(18),
        initial_ppi_weights=np.full(18, 1 / 18),
        initial_cpi_weights=np.full(18, 1 / 18),
        initial_sectoral_household_consumption=np.full(18, 200 / 18),
        initial_sectoral_firm_used_ii=np.full(18, 2.0),
        initial_total_taxes_on_products=20.0,
        initial_total_taxes_on_production=5.0,
        initial_change_in_firm_stock_inventories=10.0,
        initial_gross_fixed_capital_formation=80.0,
        initial_total_operating_surplus=260.0,
        initial_total_wages=100.0,
        initial_individual_activity=np.array([ActivityStatus.EMPLOYED, ActivityStatus.UNEMPLOYED]),
        initial_cpi_inflation=0.0,
        initial_cpi_yoy_inflation=0.0,
        initial_ppi_inflation=0.0,
        initial_hpi_inflation=0.0,
        initial_real_rent_paid=np.array([cash_rent]),
        initial_imp_rent_paid=np.array([housing]),
        initial_hh_rental_income=np.array([30.0]),
        initial_hh_consumption=200.0,
        initial_gov_consumption=50.0,
        initial_cg_rent_received=40.0,
        initial_cg_taxes_rental_income=0.0,
        initial_imports=np.array([30.0]),
        initial_imports_by_country={},
        initial_exports=np.array([40.0]),
        initial_exports_by_country={},
        export_taxes=0.0,
        initial_total_growth=0.0,
        initial_npl_ratio=0.0,
        initial_real_gross_output=371.0,
        initial_potential_output=371.0,
        initial_output_gap=0.0,
    )


@pytest.mark.parametrize("housing", [0.0, 80.0])
@pytest.mark.parametrize("adjust", [False, True])
def test_initial_and_first_period_basis_and_hdf5(test_economy, tmp_path, housing, adjust):
    test_economy.ts = initial_accounts(housing)
    ts = test_economy.ts
    for name in ("gdp_output", "gdp_expenditure", "gdp_income"):
        np.testing.assert_allclose(ts.initial(name), [450 + housing])
    np.testing.assert_allclose(ts.initial("total_household_fce"), [300 + housing])
    for name in BRIDGES:
        np.testing.assert_allclose(ts.initial(name), [housing])
    # Identical goods/cash-rent accounts, but next-period H rises by 20.
    inputs = deepcopy(CASES[0]["inputs"])
    inputs.update(rent_imputed=housing + 20, always_adjust=adjust)
    for name in ("sectoral_sales", "sectoral_intermediate_consumption"):
        inputs[name] = np.asarray(inputs[name])
    test_economy.compute_gdp(**inputs)
    for name in ("gdp_output", "gdp_expenditure", "gdp_income"):
        np.testing.assert_allclose(ts.current(name), [470 + housing])
        np.testing.assert_allclose(ts.current(name + "_growth"), [20 / (450 + housing)])
    np.testing.assert_allclose(ts.current("total_household_fce_growth"), [20 / (300 + housing)])
    for name in BRIDGES:
        np.testing.assert_allclose(ts.current(name), [housing + 20])
    with h5py.File(tmp_path / "accounting.h5", "w") as handle:
        test_economy.save_to_h5(handle.create_group("FRA"))
    with h5py.File(tmp_path / "accounting.h5", "r") as handle:
        for name in BRIDGES:
            np.testing.assert_allclose(handle["FRA/economy/" + name], [[housing], [housing + 20]])


@pytest.mark.parametrize("rent,imputed", [(-100.0, -80.0), (-100.0, 20.0), (100.0, -80.0)])
def test_initial_housing_flows_use_the_household_zero_floor(rent, imputed):
    ts = initial_accounts(imputed, cash_rent=rent)
    expected_rent = max(rent, 0.0)
    expected_imputed = max(imputed, 0.0)
    np.testing.assert_allclose(ts.initial("total_real_rent_paid"), [expected_rent])
    np.testing.assert_allclose(ts.initial("total_imp_rent_paid"), [expected_imputed])
    np.testing.assert_allclose(ts.initial("total_household_fce"), [200.0 + expected_rent + expected_imputed])
    np.testing.assert_allclose(ts.initial("gdp_output"), [350.0 + expected_rent + expected_imputed])


def test_runtime_housing_aggregates_use_the_household_zero_floor(test_economy):
    test_economy.compute_rental_market_aggregates(
        real_rent_paid=np.array([-10.0, 30.0]),
        imp_rent_paid=np.array([-20.0, 40.0]),
        rental_income=np.array([5.0, 6.0]),
    )
    np.testing.assert_allclose(test_economy.ts.current("total_real_rent_paid"), [30.0])
    np.testing.assert_allclose(test_economy.ts.current("total_imp_rent_paid"), [40.0])


@pytest.mark.parametrize("adjust", [False, True])
def test_housing_bridge_preserves_unbalanced_accounts_and_all_inputs(test_economy, adjust):
    inputs = deepcopy(CASES[0]["inputs"])
    inputs.update(rent_imputed=0.0, always_adjust=adjust, exports=77.5, operating_surplus=283.0)
    for name in ("sectoral_sales", "sectoral_intermediate_consumption"):
        inputs[name] = np.asarray(inputs[name])
    original_inputs = deepcopy(inputs)
    test_economy.compute_gdp(**inputs)
    ts = test_economy.ts
    names = (
        "gdp_output",
        "gdp_expenditure",
        "gdp_income",
        "total_household_fce",
        "total_exports",
        "total_imports",
        "total_output",
        "total_gross_value_added",
        "total_gross_operating_surplus_and_mixed_income",
        "gdp_expenditure_prebalancing_residual",
    )
    before = {name: np.asarray(ts.current(name)).copy() for name in names}
    inputs["rent_imputed"] = 80.0
    test_economy.compute_gdp(**inputs)
    for name in names:
        delta = 80 if name in ("gdp_output", "gdp_expenditure", "gdp_income", "total_household_fce") else 0
        np.testing.assert_allclose(ts.current(name), before[name] + delta)
    assert ts.current("gdp_expenditure_prebalancing_residual")[0] == pytest.approx(37.5)
    assert ts.current("gdp_income")[0] - ts.current("gdp_output")[0] == pytest.approx(23.0)
    for key in inputs:
        if key != "rent_imputed":
            np.testing.assert_array_equal(inputs[key], original_inputs[key])


@pytest.mark.parametrize("anchor", [0.0, 300.0, 600.0])
def test_proportional_initial_anchor_preserves_shares_and_source(anchor):
    source = np.array([100.0, 200.0, 0.0])
    actual = align_initial_goods_consumption(source, anchor)
    np.testing.assert_allclose(actual, [anchor / 3, 2 * anchor / 3, 0])
    np.testing.assert_array_equal(source, [100, 200, 0])
    assert actual.sum() == pytest.approx(anchor)


@pytest.mark.parametrize("goods,anchor", [([0, 0], 1), ([-1, 2], 1), ([np.nan], 1), ([1], np.inf), ([1], -1)])
def test_invalid_initial_anchor_fails_explicitly(goods, anchor):
    with pytest.raises(ValueError):
        align_initial_goods_consumption(np.asarray(goods), anchor)


def test_zero_initial_anchor_and_households():
    np.testing.assert_array_equal(align_initial_goods_consumption(np.zeros(2), 0.0), [0, 0])
