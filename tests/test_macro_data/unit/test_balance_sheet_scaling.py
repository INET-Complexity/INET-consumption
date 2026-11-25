"""Household balance-sheet scaling by k_i = y_model,i / y_HFCS,i (spec 2026-11-24)."""

import numpy as np
import pandas as pd
import pytest

from macro_data.configuration.dataconfiguration import HouseholdConsumptionInitialisationConfiguration
from macro_data.processing.synthetic_population.balance_sheet_scaling import (
    HFCS_GROSS_COLUMNS,
    HOUSEHOLD_STOCK_COLUMNS,
    apply_financial_asset_yield,
    hfcs_net_income,
    household_income_factors,
    is_scaled,
    model_income,
    scale_household_balance_sheet,
    scale_housing_market,
)
from macro_data.processing.synthetic_population.hfcs_synthetic_population import SyntheticHFCSPopulation

SCALE = 10
YEARLY = 4.0
TAX = 0.2
SIW = 0.1
ANCHOR = 3.0
TAXES = dict(financial_income_tax=TAX, employee_social_insurance_tax=SIW)


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


def set_hfcs_gross(pop, k):
    """Survey gross employee income such that model/HFCS = k on the geometric-average concept."""
    target = model_income(pop.household_data, "disp_geom_avg_income", YEARLY) / (np.asarray(k) * SCALE)
    for column in HFCS_GROSS_COLUMNS:
        pop.household_data[column] = 0.0
    pop.household_data["HFCS Gross Employee Income"] = np.nan_to_num(target) / ((1 - SIW) * (1 - TAX))
    return target


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
    hfcs_y = set_hfcs_gross(pop, k)
    before = pop.household_data.copy()

    scaled_housing, diag = scale_household_balance_sheet(
        pop, housing_table(), concept="disp_geom_avg_income", financial_asset_yield=0.0293, **TAXES
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
    hfcs_scaled = hfcs_y[:2] * SCALE
    model_y = hfcs_scaled * k[:2]
    np.testing.assert_allclose(
        after["Wealth in Deposits"].to_numpy()[:2] / model_y, before["Wealth in Deposits"].to_numpy()[:2] / hfcs_scaled
    )
    # Totals recomputed; rents are national-accounts flows and stay unchanged.
    np.testing.assert_allclose(after["Net Wealth"], after["Wealth"] - after["Debt"])
    np.testing.assert_allclose(after["Rent Paid"], before["Rent Paid"])
    np.testing.assert_allclose(after["Rent Imputed"], before["Rent Imputed"])
    # Housing: values x owner k (social unchanged); rents unchanged.
    np.testing.assert_allclose(scaled_housing["Value"], [20000.0 * k[0], 8000.0 * k[1], 5000.0])
    np.testing.assert_allclose(scaled_housing["Rent"], [300.0, 200.0, 100.0])


def test_financial_income_follows_yield_and_enters_income_net_of_tax():
    pop = toy_population()
    set_hfcs_gross(pop, [1.3, 0.9, 1.0])
    scale_household_balance_sheet(pop, None, concept="disp_geom_avg_income", financial_asset_yield=0.0293, **TAXES)
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
    set_hfcs_gross(pop, [1.3, 0.9, 1.0])
    kwargs = dict(concept="disp_geom_avg_income", financial_asset_yield=0.0293, **TAXES)
    scale_household_balance_sheet(pop, None, **kwargs)
    assert is_scaled(pop)
    deposits = pop.household_data["Wealth in Deposits"].copy()
    with pytest.raises(ValueError, match="already scaled"):
        scale_household_balance_sheet(pop, None, **kwargs)
    pop.household_data["Income from Financial Assets"] = 0.0
    apply_financial_asset_yield(pop, 0.0293, financial_income_tax=TAX)
    np.testing.assert_allclose(pop.household_data["Wealth in Deposits"], deposits)
    assert pop.household_data["Income from Financial Assets"].iloc[0] > 0


def test_missing_hfcs_gross_income_or_yield_is_an_error():
    pop = toy_population()
    with pytest.raises(ValueError, match="HFCS gross income"):
        scale_household_balance_sheet(pop, None, concept="disp_geom_avg_income", financial_asset_yield=0.03, **TAXES)
    set_hfcs_gross(pop, [1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="yield"):
        scale_household_balance_sheet(pop, None, concept="disp_geom_avg_income", financial_asset_yield=0.0, **TAXES)


def test_hfcs_net_income_uses_runtime_taxes_and_leaves_transfers_untaxed():
    data = pd.DataFrame({c: [0.0] for c in HFCS_GROSS_COLUMNS})
    data.loc[0, ["HFCS Gross Employee Income", "HFCS Gross Self-Employment Income"]] = [800.0, 200.0]
    data.loc[0, ["HFCS Gross Pension Income", "HFCS Gross Social Transfers"]] = [300.0, 100.0]
    data.loc[0, ["HFCS Gross Rental Income", "HFCS Gross Financial Income"]] = [50.0, 50.0]
    taxes = dict(income_tax=0.2, employee_social_insurance_tax=0.1)
    non_property = 1000.0 * 0.9 * 0.8 + 400.0
    total = non_property + 100.0 * 0.8
    np.testing.assert_allclose(hfcs_net_income(data, "disp_non_property_income", **taxes), [non_property])
    np.testing.assert_allclose(hfcs_net_income(data, "disposable_income", **taxes), [total])
    np.testing.assert_allclose(hfcs_net_income(data, "disp_geom_avg_income", **taxes), [np.sqrt(total * non_property)])


def test_housing_rows_without_household_owner_are_unchanged():
    out = scale_housing_market(housing_table(), np.array([2.0, 3.0]))
    np.testing.assert_allclose(out["Value"], [40000.0, 24000.0, 5000.0])
    np.testing.assert_allclose(out["Rent"], [300.0, 200.0, 100.0])


def test_configuration_defaults_and_validation():
    default = HouseholdConsumptionInitialisationConfiguration()
    assert default.balance_sheet_scaling is False
    assert default.financial_asset_yield == pytest.approx(0.0293)
    assert default.scaling_income_concept == "disp_geom_avg_income"
    assert HouseholdConsumptionInitialisationConfiguration(balance_sheet_scaling=True).balance_sheet_scaling
    with pytest.raises(ValueError, match="financial_asset_yield"):
        HouseholdConsumptionInitialisationConfiguration(balance_sheet_scaling=True, financial_asset_yield=0.0)


def matched_population_and_housing():
    """Household 0 lives in its home (house 0); household 1 lets house 1 to household 2; house 2 is social."""
    pop = toy_population()
    data = pop.household_data
    data["Tenure Status of the Main Residence"] = [1, 1, 3]
    data["Corresponding Inhabited House ID"] = [0.0, np.nan, 1.0]
    housing = pd.DataFrame(
        {
            "House ID": [0, 1, 2],
            "Corresponding Owner Household ID": [0, 1, -1],
            "Corresponding Inhabitant Household ID": [0.0, 2.0, np.nan],
            "Is Owner-Occupied": [True, False, False],
            "Value": [20000.0, 15000.0, 5000.0],
            "Rent": [300.0, 200.0, 100.0],
        }
    )
    set_hfcs_gross(pop, [1.3, 1.2, 1.0])
    return pop, housing


def test_imputed_rent_total_is_unchanged_and_equals_the_owner_dwelling_rent():
    pop, housing = matched_population_and_housing()
    imputed_before = pop.household_data["Rent Imputed"].sum()
    scaled, _ = scale_household_balance_sheet(
        pop, housing, concept="disp_geom_avg_income", financial_asset_yield=0.0293, **TAXES
    )
    data = pop.household_data
    k = data["Balance Sheet Scaling Factor"].to_numpy()
    assert data["Rent Imputed"].sum() == pytest.approx(imputed_before)
    assert data.loc[0, "Rent Imputed"] == pytest.approx(scaled.loc[0, "Rent"])
    np.testing.assert_allclose(scaled["Value"], [20000.0 * k[0], 15000.0 * k[1], 5000.0])


def test_rent_floor_is_reapplied_after_scaling_and_never_lowers_a_rent():
    pop, housing = matched_population_and_housing()
    before = pop.household_data.copy()
    phi = 0.05  # per quarter 0.0125 x value: not binding before scaling (250 < 300, 187.5 < 200)
    scaled, diag = scale_household_balance_sheet(
        pop,
        housing,
        concept="disp_geom_avg_income",
        financial_asset_yield=0.0293,
        minimum_annual_rental_yield=phi,
        rental_income_taxes=TAX,
        **TAXES,
    )
    data = pop.household_data
    k = data["Balance Sheet Scaling Factor"].to_numpy()
    floor = phi * 3 / 12 * scaled["Value"].to_numpy()
    expected_rent = np.maximum(housing["Rent"].to_numpy(), floor)
    np.testing.assert_allclose(scaled["Rent"], expected_rent)
    assert np.all(scaled["Rent"].to_numpy() >= housing["Rent"].to_numpy())
    # Owner: imputed rent follows its dwelling to the floor.
    assert scaled.loc[0, "Rent"] == pytest.approx(phi / 4 * 20000.0 * k[0])
    assert data.loc[0, "Rent Imputed"] == pytest.approx(scaled.loc[0, "Rent"])
    # Rented dwelling: tenant pays the raise; landlord receives it net of the rental income tax.
    raise_rented = scaled.loc[1, "Rent"] - 200.0
    assert raise_rented == pytest.approx(phi / 4 * 15000.0 * k[1] - 200.0)
    assert data.loc[2, "Rent Paid"] == pytest.approx(before.loc[2, "Rent Paid"] + raise_rented)
    assert data.loc[1, "Rental Income from Real Estate"] == pytest.approx(
        before.loc[1, "Rental Income from Real Estate"] + (1 - TAX) * raise_rented
    )
    assert scaled.loc[2, "Rent"] == pytest.approx(100.0)  # social dwelling above its floor: unchanged
    assert diag["rent_floor_binding_owner_occupied"] == 1
    assert diag["rent_floor_binding_rented"] == 1
    assert diag["rent_floor_uplift_rented"] == pytest.approx(raise_rented)
