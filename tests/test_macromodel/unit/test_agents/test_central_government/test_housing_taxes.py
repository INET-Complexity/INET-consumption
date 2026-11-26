"""Household housing taxes: property tax (owner, on value) and residence tax (occupant, on rental value)."""

import numpy as np
import pytest

PERIODS_PER_YEAR = 4


def _bases(households, housing_market):
    properties = housing_market.states["properties"]
    return {
        "property_values": properties["Value"].fillna(0.0).values,
        "property_owner_ids": properties["Corresponding Owner Household ID"].values,
        "occupant_rental_value": households.ts.current("rent") + households.ts.current("rent_imputed"),
    }


def _calibrate(central_government, bases, property_target, residence_target):
    central_government.calibrate_housing_taxes(
        property_tax_revenue_target_annual=property_target,
        residence_tax_revenue_target_annual=residence_target,
        periods_per_year=PERIODS_PER_YEAR,
        **bases,
    )


def _settle(central_government, property_tax, residence_tax):
    central_government.compute_taxes(
        current_ind_employee_income=np.zeros(1),
        current_total_rent_paid=0.0,
        current_income_financial_assets=np.zeros(1),
        current_ind_activity=np.zeros(1, dtype=int),
        current_ind_realised_cons=np.zeros(1),
        current_bank_profits=np.zeros(1),
        current_firm_production=np.zeros(1),
        current_firm_price=np.zeros(1),
        current_firm_profits=np.zeros(1),
        current_firm_industries=np.zeros(1, dtype=int),
        current_household_new_real_wealth=np.zeros(1),
        taxes_less_subsidies_rates=np.zeros(1),
        current_total_exports=0.0,
        household_property_taxes=property_tax.sum(),
        household_residence_taxes=residence_tax.sum(),
    )


def test_t0_revenue_equals_annual_target(test_central_government, test_households, test_housing_market):
    bases = _bases(test_households, test_housing_market)
    _calibrate(test_central_government, bases, 1.9752e10, 1.84e10)
    property_tax, residence_tax = test_central_government.compute_household_housing_taxes(**bases)
    assert PERIODS_PER_YEAR * property_tax.sum() == pytest.approx(1.9752e10, rel=1e-12)
    assert PERIODS_PER_YEAR * residence_tax.sum() == pytest.approx(1.84e10, rel=1e-12)


def test_household_payments_equal_government_receipts(test_central_government, test_households, test_housing_market):
    bases = _bases(test_households, test_housing_market)
    income_before = test_households.compute_income()
    _calibrate(test_central_government, bases, 1.9752e10, 1.84e10)
    property_tax, residence_tax = test_central_government.compute_household_housing_taxes(**bases)
    test_households.ts.override_current("housing_taxes_property", property_tax)
    test_households.ts.override_current("housing_taxes_residence", residence_tax)
    _settle(test_central_government, property_tax, residence_tax)

    paid = income_before.sum() - test_households.compute_income().sum()
    received = (
        test_central_government.ts.current("taxes_property")[0]
        + test_central_government.ts.current("taxes_residence")[0]
    )
    assert paid == pytest.approx(received, rel=1e-12)
    assert test_central_government.compute_revenue(household_rent_paid_to_government=0.0) == pytest.approx(
        received, rel=1e-12
    )


def test_zero_when_disabled(test_central_government, test_households, test_housing_market):
    bases = _bases(test_households, test_housing_market)
    _calibrate(test_central_government, bases, None, None)
    property_tax, residence_tax = test_central_government.compute_household_housing_taxes(**bases)
    assert not property_tax.any() and not residence_tax.any()
    _settle(test_central_government, property_tax, residence_tax)
    assert test_central_government.ts.current("taxes_property")[0] == 0.0
    assert test_central_government.ts.current("taxes_residence")[0] == 0.0


def test_owner_pays_on_all_owned_dwellings_and_occupant_pays_residence_tax(test_central_government):
    # Household 0 owns and occupies dwelling 0 and rents dwelling 1 out to household 1.
    # Dwelling 2 has no household owner (social housing); household 2 is its tenant.
    bases = {
        "property_values": np.array([300.0, 100.0, 50.0]),
        "property_owner_ids": np.array([0, 0, -1]),
        "occupant_rental_value": np.array([12.0, 4.0, 2.0]),
    }
    _calibrate(test_central_government, bases, 4.0 * 400.0 * 0.01, 4.0 * 18.0 * 0.1)
    property_tax, residence_tax = test_central_government.compute_household_housing_taxes(**bases)
    np.testing.assert_allclose(property_tax, [4.0, 0.0, 0.0])
    np.testing.assert_allclose(residence_tax, [1.2, 0.4, 0.2])
