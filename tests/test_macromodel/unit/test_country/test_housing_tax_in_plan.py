"""Housing taxes are fixed before planning and enter the plan's cash resources."""

import numpy as np


def _set_housing_tax_rates(country, property_rate: float, residence_rate: float) -> None:
    country.central_government.states["Property Tax Rate"] = property_rate
    country.central_government.states["Residence Tax Rate"] = residence_rate


def _expected_taxes(country) -> np.ndarray:
    properties = country.housing_market.states["properties"]
    property_tax, residence_tax = country.central_government.compute_household_housing_taxes(
        property_values=properties["Value"].fillna(0.0).values,
        property_owner_ids=properties["Corresponding Owner Household ID"].values,
        occupant_rental_value=country.households.ts.current("rent") + country.households.ts.current("rent_imputed"),
    )
    return property_tax + residence_tax


def test_housing_taxes_are_fixed_from_start_of_period_owner_and_occupant(test_country):
    _set_housing_tax_rates(test_country, 0.002, 0.05)
    expected = _expected_taxes(test_country)
    assert expected.sum() > 0.0

    test_country._append_household_housing_taxes()

    np.testing.assert_allclose(test_country._current_household_housing_taxes(), expected)


def test_financing_income_is_net_of_housing_taxes(test_country):
    _set_housing_tax_rates(test_country, 0.002, 0.05)

    test_country._set_household_income_expectations(replace_current=False)

    hh = test_country.households
    taxes = test_country._current_household_housing_taxes()
    assert taxes.sum() > 0.0
    np.testing.assert_allclose(hh.ts.current("financing_income"), hh.ts.current("expected_income") - taxes)
    # Before known receipts are refreshed, the fallback is net of the same taxes.
    hh._known_receipts_ready = False
    np.testing.assert_allclose(hh.current_financing_income(), hh.ts.current("expected_income") - taxes)


def test_known_receipts_refresh_keeps_housing_taxes_in_the_plan(test_country):
    _set_housing_tax_rates(test_country, 0.0, 0.0)
    test_country._set_household_income_expectations(replace_current=False)
    test_country.refresh_known_household_receipts()
    untaxed = np.asarray(test_country.households.current_financing_income(), dtype=float).copy()

    _set_housing_tax_rates(test_country, 0.002, 0.05)
    test_country._append_household_housing_taxes()
    test_country.refresh_known_household_receipts()

    taxes = test_country._current_household_housing_taxes()
    assert taxes.sum() > 0.0
    np.testing.assert_allclose(test_country.households.current_financing_income(), untaxed - taxes)


def test_realised_income_uses_the_planned_housing_taxes(test_country):
    _set_housing_tax_rates(test_country, 0.002, 0.05)
    test_country._append_household_housing_taxes()
    hh = test_country.households
    taxes = test_country._current_household_housing_taxes()

    _set_housing_tax_rates(test_country, 0.0, 0.0)  # a later rate change must not alter this period's tax
    financial = (
        hh.ts.current("income_dividend_distributions")
        if getattr(hh.functions["wealth"], "illiquid_returns_are_capital_gains", False)
        else hh.ts.current("income_financial_assets")
    )
    gross = (
        hh.ts.current("income_employee")
        + hh.ts.current("income_social_transfers")
        + hh.ts.current("income_rental")
        + financial
    )

    np.testing.assert_allclose(hh.compute_income(), gross - taxes)
    np.testing.assert_allclose(test_country._current_household_housing_taxes(), taxes)
