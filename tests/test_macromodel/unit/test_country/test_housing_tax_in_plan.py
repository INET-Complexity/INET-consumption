"""Housing taxes are fixed once at the start of the period and used unchanged by the plan and the settlement."""

from copy import deepcopy

import numpy as np
import pytest

from tests.test_macromodel.unit.test_country.test_minimum_income_entitlement import _production_simulation

PROPERTY_RATE, RESIDENCE_RATE = 0.002, 0.05
OWNER = "Corresponding Owner Household ID"
TAX_SERIES = ("housing_taxes_property", "housing_taxes_residence")
SETTLEMENT_SERIES = (
    "liquid_financial_assets",
    "illiquid_financial_assets",
    "portfolio_settlement_committed_adjustment_cost",
    "stage5_cash_ledger_residual",
)


def _set_housing_tax_rates(country, property_rate: float, residence_rate: float) -> None:
    country.central_government.states["Property Tax Rate"] = property_rate
    country.central_government.states["Residence Tax Rate"] = residence_rate


def _change_owners_and_values(country) -> None:
    """Move every owned dwelling to the next owner and raise every value by half."""
    properties = country.housing_market.states["properties"]
    owners = properties[OWNER].to_numpy(copy=True)
    owned = owners >= 0
    owners[owned] = np.roll(owners[owned], 1)
    properties[OWNER] = owners
    properties["Value"] = 1.5 * properties["Value"]


def _recomputed_taxes(country) -> dict[str, np.ndarray]:
    """What a recomputation from the current inputs would charge (the pre-#261 settlement rule)."""
    properties = country.housing_market.states["properties"]
    hh = country.households
    property_tax, residence_tax = country.central_government.compute_household_housing_taxes(
        property_values=properties["Value"].fillna(0.0).values,
        property_owner_ids=properties[OWNER].values,
        occupant_rental_value=hh.ts.current("rent") + hh.ts.current("rent_imputed"),
    )
    return dict(zip(TAX_SERIES, (property_tax, residence_tax)))


def _materially_different(recomputed: np.ndarray, fixed: np.ndarray) -> bool:
    return np.abs(recomputed - fixed).sum() > 0.1 * fixed.sum() > 0.0


def test_expectations_step_fixes_housing_taxes_once_per_period(test_country):
    _set_housing_tax_rates(test_country, PROPERTY_RATE, RESIDENCE_RATE)
    hh = test_country.households
    realised_rows = len(hh.ts.historic("income"))

    test_country._set_household_income_expectations(replace_current=False)
    fixed = hh.current_housing_taxes().copy()
    fixed_property = hh.ts.current("housing_taxes_property").copy()
    _change_owners_and_values(test_country)
    assert _materially_different(_recomputed_taxes(test_country)["housing_taxes_property"], fixed_property)
    test_country._set_household_income_expectations(replace_current=False)
    test_country._set_household_income_expectations(replace_current=True)

    # One row for this period: the series stays one row ahead of realised income until settlement.
    for name in TAX_SERIES:
        assert len(hh.ts.historic(name)) == realised_rows + 1
    np.testing.assert_array_equal(hh.current_housing_taxes(), fixed)
    np.testing.assert_array_equal(hh.ts.current("housing_taxes_property"), fixed_property)


def test_financing_income_and_its_fallback_are_net_of_housing_taxes(test_country):
    untaxed = deepcopy(test_country)
    _set_housing_tax_rates(untaxed, 0.0, 0.0)
    _set_housing_tax_rates(test_country, PROPERTY_RATE, RESIDENCE_RATE)

    for country in (untaxed, test_country):
        country._set_household_income_expectations(replace_current=False)

    hh, hh_untaxed = test_country.households, untaxed.households
    taxes = hh.current_housing_taxes()
    assert taxes.sum() > 0.0
    np.testing.assert_array_equal(hh_untaxed.current_housing_taxes(), 0.0)
    # Expected income (property demand, credit targets, consumption) is unchanged; financing income nets the tax.
    np.testing.assert_allclose(hh.ts.current("expected_income"), hh_untaxed.ts.current("expected_income"))
    np.testing.assert_allclose(
        hh.ts.current("financing_income"), hh_untaxed.ts.current("financing_income") - taxes, rtol=1e-12, atol=1e-6
    )
    assert not hh._known_receipts_ready
    np.testing.assert_allclose(
        hh.current_financing_income(), hh_untaxed.current_financing_income() - taxes, rtol=1e-12, atol=1e-6
    )


def test_known_receipts_refresh_keeps_housing_taxes_in_the_plan(test_country):
    untaxed = deepcopy(test_country)
    _set_housing_tax_rates(untaxed, 0.0, 0.0)
    _set_housing_tax_rates(test_country, PROPERTY_RATE, RESIDENCE_RATE)

    for country in (untaxed, test_country):
        country._set_household_income_expectations(replace_current=False)
        country._set_household_income_expectations(replace_current=True)
        country.refresh_known_household_receipts()

    taxes = test_country.households.current_housing_taxes()
    assert taxes.sum() > 0.0
    np.testing.assert_allclose(
        test_country.households.current_financing_income(),
        untaxed.households.current_financing_income() - taxes,
        rtol=1e-12,
        atol=1e-6,
    )


@pytest.fixture(scope="module")
def settled_period(datawrapper):
    """One production period with housing taxes on, owners and values changed where the tax used to be computed.

    Before #261 the tax was recomputed in update_realised_metrics right after reconcile_realised_household_support.
    The wrapper changes owners, values and the residence rate at that point; the inputs are restored after
    update_realised_metrics so the rest of the period is unaffected. At the wealth update, a copy of the households
    is settled on income gross of the tax, for the asset-side SFC check.
    """
    simulation = _production_simulation(datawrapper, {"mode": "minimum_income"})
    country = simulation.countries["FRA"]
    hh, government = country.households, country.central_government
    _set_housing_tax_rates(country, PROPERTY_RATE, RESIDENCE_RATE)
    record = {}
    append, reconcile, realise = (
        country._append_household_housing_taxes,
        country.reconcile_realised_household_support,
        country.update_realised_metrics,
    )

    def append_and_record():
        append()
        record["fixed"] = {name: hh.ts.current(name).copy() for name in TAX_SERIES}
        record["rows_at_fix"] = len(hh.ts.historic(TAX_SERIES[0]))

    def reconcile_after_input_change():
        properties = country.housing_market.states["properties"]
        record["saved_inputs"] = (properties[[OWNER, "Value"]].copy(), government.states["Residence Tax Rate"])
        _change_owners_and_values(country)
        government.states["Residence Tax Rate"] = 2.0 * RESIDENCE_RATE
        record["recomputed"] = _recomputed_taxes(country)
        reconcile()

    def realise_and_restore(*args, **kwargs):
        try:
            return realise(*args, **kwargs)
        finally:
            saved, residence_rate = record.pop("saved_inputs")
            properties = country.housing_market.states["properties"]
            properties[[OWNER, "Value"]] = saved
            government.states["Residence Tax Rate"] = residence_rate

    def update_wealth_and_untaxed_copy(*args, **kwargs):
        # Counterfactual settlement: the same household and flows, with realised income gross of the housing tax.
        # Country.__deepcopy__ handles the TimeSeries copy protocol; a bare Households deepcopy does not.
        untaxed = deepcopy(country).households
        untaxed.ts.override_current("income", hh.ts.current("income") + hh.current_housing_taxes())
        type(hh).update_wealth(untaxed, *args, **kwargs)
        record["untaxed"] = {name: untaxed.ts.current(name).copy() for name in SETTLEMENT_SERIES}
        return update_wealth(*args, **kwargs)

    update_wealth = hh.update_wealth
    country._append_household_housing_taxes = append_and_record
    country.reconcile_realised_household_support = reconcile_after_input_change
    country.update_realised_metrics = realise_and_restore
    hh.update_wealth = update_wealth_and_untaxed_copy
    simulation.iterate(t=0)
    return country, record


def test_settlement_charges_the_start_of_period_housing_taxes(settled_period):
    country, record = settled_period
    hh = country.households
    fixed = record["fixed"]
    for name in TAX_SERIES:
        # The changed inputs would have produced a different tax: settlement must not have used them.
        assert _materially_different(record["recomputed"][name], fixed[name])
        np.testing.assert_array_equal(hh.ts.current(name), fixed[name])
        # No row added after the fix; one row per period, aligned with realised income again.
        assert len(hh.ts.historic(name)) == record["rows_at_fix"] == len(hh.ts.historic("income"))


def test_household_housing_tax_debit_equals_government_revenue(settled_period):
    country, record = settled_period
    government = country.central_government
    fixed = record["fixed"]
    for series, revenue in (
        ("housing_taxes_property", "taxes_property"),
        ("housing_taxes_residence", "taxes_residence"),
    ):
        debit = float(fixed[series].sum())
        assert debit > 0.0
        np.testing.assert_allclose(government.ts.current(revenue)[0], debit, rtol=1e-12)


def test_household_financial_assets_carry_the_housing_tax_debit(settled_period):
    """Asset side of the SFC check: the tax leaves household deposits, not only the income accounts.

    Compared with the same settlement on income gross of the tax, closing liquid plus illiquid financial assets are
    lower by exactly the tax, once the portfolio adjustment cost (which depends on the wealth being rebalanced) is
    netted. The saving rule splits the debit between deposits and illiquid purchases, so deposits alone carry most
    of it, not all.
    """
    country, record = settled_period
    hh = country.households
    taxes = sum(record["fixed"][name] for name in TAX_SERIES)
    assert taxes.sum() > 0.0
    diff = {name: hh.ts.current(name) - record["untaxed"][name] for name in SETTLEMENT_SERIES}
    np.testing.assert_allclose(
        diff["liquid_financial_assets"]
        + diff["illiquid_financial_assets"]
        + diff["portfolio_settlement_committed_adjustment_cost"],
        -taxes,
        rtol=1e-9,
        atol=1e-3,
    )
    assert diff["liquid_financial_assets"].sum() < 0.0
    # The realised cash ledger closes in both settlements: the tax is a cash use, not a ledger residual.
    for residual in (hh.ts.current("stage5_cash_ledger_residual"), record["untaxed"]["stage5_cash_ledger_residual"]):
        np.testing.assert_allclose(residual, 0.0, atol=1e-7)
