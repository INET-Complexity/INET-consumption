"""Household balance-sheet scaling by k_i = y_model,i / y_HFCS,i (spec 2026-11-24)."""

import numpy as np
import pandas as pd
import pytest

from macro_data.configuration.dataconfiguration import HouseholdConsumptionInitialisationConfiguration
from macro_data.processing.synthetic_population.balance_sheet_scaling import (
    HOUSEHOLD_STOCK_COLUMNS,
    apply_financial_asset_yield,
    household_income_factors,
    is_scaled,
    model_income,
    scale_household_balance_sheet,
    scale_housing_market,
)
from macro_data.processing.synthetic_population.hfcs_synthetic_population import SyntheticHFCSPopulation
from tests.test_macro_data.unit.test_hfcs_consumption_source import save_read, source_frame

SCALE = 10
YEARLY = 4.0
TAX = 0.2
ANCHOR = 3.0


def toy_population():
    """Three households; quarterly model flows; stocks already x SCALE; IFA x ANCHOR (national-accounts anchor)."""
    pop = SyntheticHFCSPopulation.__new__(SyntheticHFCSPopulation)
    pop.scale = SCALE
    pop.yearly_factor = YEARLY
    pop.ifa_anchor_factor = ANCHOR
    data = pd.DataFrame(
        {
            "Employee Income": [3000.0, 1000.0, 0.0],
            "Regular Social Transfers": [0.0, 500.0, 0.0],
            "Rental Income from Real Estate": [0.0, 250.0, 0.0],
            "Income from Financial Assets": [100.0, 50.0, 0.0],
            "Wealth in Deposits": [1000.0, 2000.0, 50.0],
            "Wealth in Other Financial Assets": [3000.0 * ANCHOR, 0.0, 0.0],
            "Outstanding Balance of HMR Mortgages": [5000.0, 0.0, 0.0],
            "Outstanding Balance of Mortgages on other Properties": [0.0, 1000.0, 0.0],
            "Outstanding Balance of other Non-Mortgage Loans": [100.0, 0.0, 10.0],
            "Value of the Main Residence": [20000.0, 0.0, 0.0],
            "Value of other Properties": [0.0, 8000.0, 0.0],
            "Wealth Other Real Assets": [500.0, 100.0, 0.0],
            "Rent Imputed": [300.0, 0.0, 0.0],
            "Rent Paid": [0.0, 0.0, 200.0],
        }
    )
    pop.household_data = data
    pop.set_household_income(financial_income_tax=TAX)
    return pop


def hfcs_income(pop, k):
    """HFCS per-household annual income such that model/HFCS = k on the geometric-average concept."""
    model = model_income(pop.household_data, "disp_geom_avg_income", YEARLY)
    return model / (np.asarray(k) * SCALE)


def housing_table():
    return pd.DataFrame(
        {
            "Corresponding Owner Household ID": [0, 1, -1],
            "Is Owner-Occupied": [True, False, False],
            "Value": [20000.0, 8000.0, 5000.0],
            "Rent": [300.0, 200.0, 100.0],
        }
    )


def test_model_income_concepts_are_annual_and_consistent():
    pop = toy_population()
    data = pop.household_data
    total = data["Income"].to_numpy() * YEARLY
    non_property = (data["Employee Income"] + data["Regular Social Transfers"]).to_numpy() * YEARLY
    np.testing.assert_allclose(model_income(data, "disposable_income", YEARLY), total)
    np.testing.assert_allclose(model_income(data, "disp_non_property_income", YEARLY), non_property)
    np.testing.assert_allclose(
        model_income(data, "disp_geom_avg_income", YEARLY)[:2], np.sqrt(total * non_property)[:2]
    )
    with pytest.raises(ValueError, match="Unknown"):
        model_income(data, "gross_income", YEARLY)


def test_factors_fall_back_to_common_factor_for_nonpositive_income():
    factors, n_fallback = household_income_factors(np.array([200.0, 100.0, 0.0]), np.array([100.0, 100.0, 50.0]))
    np.testing.assert_allclose(factors, [2.0, 1.0, 1.5])
    assert n_fallback == 1


def test_stocks_scale_by_household_factor_so_ratios_match_hfcs():
    pop = toy_population()
    k = np.array([1.3, 0.9, 1.0])
    pop.household_data["HFCS Income disp_geom_avg_income"] = hfcs_income(pop, k)
    before = pop.household_data.copy()

    scaled_housing, diag = scale_household_balance_sheet(
        pop, housing_table(), concept="disp_geom_avg_income", financial_asset_yield=0.0293, financial_income_tax=TAX
    )
    after = pop.household_data
    k_used = after["Balance Sheet Scaling Factor"].to_numpy()
    np.testing.assert_allclose(k_used[:2], k[:2])  # household 2 has zero income -> common factor
    assert diag["n_common_factor_fallback"] == 1

    for name in HOUSEHOLD_STOCK_COLUMNS:
        expected = before[name].to_numpy() * k_used
        if name == "Wealth in Other Financial Assets":
            expected = expected / ANCHOR  # restarts from the survey level
        np.testing.assert_allclose(after[name].to_numpy(), expected, err_msg=name)
    # Stock / HFCS-scaled income equals survey stock / survey income for every household.
    hfcs_scaled = after["HFCS Income disp_geom_avg_income"].to_numpy() * SCALE
    model_y = hfcs_scaled[:2] * k[:2]
    np.testing.assert_allclose(
        after["Wealth in Deposits"].to_numpy()[:2] / model_y,
        before["Wealth in Deposits"].to_numpy()[:2] / hfcs_scaled[:2],
    )
    # Totals recomputed; rent paid unchanged.
    np.testing.assert_allclose(after["Net Wealth"], after["Wealth"] - after["Debt"])
    np.testing.assert_allclose(after["Rent Paid"], before["Rent Paid"])
    # Housing: owner-occupied value and rent x owner k; rented value x landlord k, rent unchanged; social unchanged.
    np.testing.assert_allclose(scaled_housing["Value"], [20000.0 * k[0], 8000.0 * k[1], 5000.0])
    np.testing.assert_allclose(scaled_housing["Rent"], [300.0 * k[0], 200.0, 100.0])


def test_financial_income_follows_yield_and_enters_income_net_of_tax():
    pop = toy_population()
    pop.household_data["HFCS Income disp_geom_avg_income"] = hfcs_income(pop, [1.3, 0.9, 1.0])
    scale_household_balance_sheet(
        pop, None, concept="disp_geom_avg_income", financial_asset_yield=0.0293, financial_income_tax=TAX
    )
    data = pop.household_data
    np.testing.assert_allclose(
        data["Income from Financial Assets"], 0.0293 / YEARLY * data["Wealth in Other Financial Assets"]
    )
    np.testing.assert_allclose(
        data["Income"],
        data["Employee Income"]
        + data["Regular Social Transfers"]
        + data["Rental Income from Real Estate"]
        + (1 - TAX) * data["Income from Financial Assets"],
    )
    assert pop.coefficient_fa_income == pytest.approx(0.0293 / YEARLY)


def test_scaling_twice_is_refused_and_yield_can_be_reapplied():
    pop = toy_population()
    pop.household_data["HFCS Income disp_geom_avg_income"] = hfcs_income(pop, [1.3, 0.9, 1.0])
    kwargs = dict(concept="disp_geom_avg_income", financial_asset_yield=0.0293, financial_income_tax=TAX)
    scale_household_balance_sheet(pop, None, **kwargs)
    assert is_scaled(pop)
    deposits = pop.household_data["Wealth in Deposits"].copy()
    with pytest.raises(ValueError, match="already scaled"):
        scale_household_balance_sheet(pop, None, **kwargs)
    pop.household_data["Income from Financial Assets"] = 0.0
    apply_financial_asset_yield(pop, 0.0293, financial_income_tax=TAX)
    np.testing.assert_allclose(pop.household_data["Wealth in Deposits"], deposits)
    assert pop.household_data["Income from Financial Assets"].iloc[0] > 0


def test_missing_hfcs_income_or_yield_is_an_error():
    pop = toy_population()
    with pytest.raises(ValueError, match="HFCS Income"):
        scale_household_balance_sheet(
            pop, None, concept="disp_geom_avg_income", financial_asset_yield=0.03, financial_income_tax=TAX
        )
    pop.household_data["HFCS Income disp_geom_avg_income"] = 1.0
    with pytest.raises(ValueError, match="yield"):
        scale_household_balance_sheet(
            pop, None, concept="disp_geom_avg_income", financial_asset_yield=0.0, financial_income_tax=TAX
        )


def test_housing_rows_without_household_owner_are_unchanged():
    out = scale_housing_market(housing_table(), np.array([2.0, 3.0]))
    np.testing.assert_allclose(out["Value"], [40000.0, 24000.0, 5000.0])
    np.testing.assert_allclose(out["Rent"], [600.0, 200.0, 100.0])


def test_configuration_requires_hfcs_source_and_positive_yield():
    with pytest.raises(ValueError, match="hfcs_cy"):
        HouseholdConsumptionInitialisationConfiguration(balance_sheet_scaling=True, financial_asset_yield=0.03)
    source = dict(
        method="hfcs_cy",
        source_file="s.parquet",
        registration_file="r.json",
        source_sha256="a" * 64,
        registration_sha256="b" * 64,
    )
    with pytest.raises(ValueError, match="financial_asset_yield"):
        HouseholdConsumptionInitialisationConfiguration(balance_sheet_scaling=True, **source)
    config = HouseholdConsumptionInitialisationConfiguration(
        balance_sheet_scaling=True, financial_asset_yield=0.0293, scaling_income_concept="disposable_income", **source
    )
    assert config.scaling_income_concept == "disposable_income"
    assert HouseholdConsumptionInitialisationConfiguration().balance_sheet_scaling is False


def test_source_reader_exposes_optional_income_columns(tmp_path):
    frame = source_frame()
    frame["disp_geom_avg_income"] = [10.0, 20.0, 30.0, np.nan]
    source = save_read(tmp_path, frame)
    np.testing.assert_allclose(
        source.incomes_for(np.array(["003", "001", "003"]), "disp_geom_avg_income"), [30.0, 10.0, 30.0]
    )
    with pytest.raises(ValueError, match="no income column"):
        source.incomes_for(np.array(["001"]), "disposable_income")
    assert save_read(tmp_path, source_frame()).incomes == {}
