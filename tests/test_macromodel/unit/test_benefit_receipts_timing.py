"""Benefits planned as known receipts must equal benefits paid in the same period.

Known receipts (``financing_income``) are refreshed after labour-market clearing.
They must use the post-clearing unemployment-benefit vector and the benefit price
level CPI_{t-1}, the same quantities that the realised booking and the fiscal
totals use, so that planned benefit receipts equal paid benefit receipts.
"""

import numpy as np
import pytest

from macromodel.agents.individuals.individual_properties import ActivityStatus
from macromodel.configurations import CountryConfiguration, SimulationConfiguration
from macromodel.simulation import Simulation
from tests.test_macromodel.unit.test_country.test_minimum_income_entitlement import _production_simulation

UB_NAME = "income_from_unemployment_benefits"
# Expected benefit term -> paid benefit term.
EXPECTED_BENEFITS = {
    "expected_income_public_pension": "income_public_pension",
    "expected_income_unemployment_benefits": "income_unemployment_benefits",
    "expected_income_other_social_transfers": "income_other_social_transfers",
}


def _run_instrumented(simulation, periods):
    """Run the periods and record the last period's benefit vectors used for planning and for payment."""
    country = simulation.countries["FRA"]
    record = {}
    clear, refresh, components, expectations = (
        country.clear_labour_market,
        country.refresh_known_household_receipts,
        country.compute_realised_household_social_income_components,
        country._set_household_income_expectations,
    )

    def clear_labour_market():
        record["status_before"] = country.individuals.states["Activity Status"].copy()
        clear()
        record["status_after"] = country.individuals.states["Activity Status"].copy()

    def refresh_known_household_receipts():
        record["phase"] = "planned"
        refresh()
        record["planned_ub_individual"] = country.individuals.ts.current(UB_NAME).copy()
        record["planned_benefit_price_level"] = country.economy.current_consumer_price_level()
        record["phase"] = None

    def compute_components():
        result = components()
        if record.get("phase") == "planned":
            record["planned_components"] = tuple(np.copy(x) for x in result)
        return result

    def set_household_income_expectations(*, replace_current):
        expectations(replace_current=replace_current)
        ts = country.households.ts
        record[2 if replace_current else 1] = {name: ts.current(name).copy() for name in EXPECTED_BENEFITS}
        record[("wage", 2 if replace_current else 1)] = {
            "expected": ts.current("expected_income_employee").copy(),
            "locked_wages": country.households.aggregate_individual_amount(
                country.nominal_employee_income(), country.individuals.states["Corresponding Household ID"]
            ),
            "expected_individual": country.individuals.ts.current("expected_income").copy(),
            "employed": country.individuals.states["Activity Status"] == ActivityStatus.EMPLOYED,
            "nominal_individual": country.nominal_employee_income().copy(),
        }
        record["expected_inflation"] = country.economy.current_expected_consumer_period_inflation()

    country.clear_labour_market = clear_labour_market
    country._set_household_income_expectations = set_household_income_expectations
    country.refresh_known_household_receipts = refresh_known_household_receipts
    country.compute_realised_household_social_income_components = compute_components
    for period in range(periods):
        simulation.iterate(t=period)
    return country, record


@pytest.fixture(scope="module")
def instrumented_period(datawrapper):
    """Legacy mode. In this fixture the first period with both job finders and job losers is the fourth."""
    simulation = Simulation.from_datawrapper(
        datawrapper=datawrapper,
        simulation_configuration=SimulationConfiguration(
            country_configurations={"FRA": CountryConfiguration()}, seed=15
        ),
    )
    return _run_instrumented(simulation, periods=4)


@pytest.fixture(scope="module")
def instrumented_minimum_income_period(datawrapper):
    """Minimum income mode (production household rules, resolver on): T is fixed at the start of each period."""
    return _run_instrumented(_production_simulation(datawrapper, {"mode": "minimum_income"}), periods=4)


def _status_groups(record):
    before, after = record["status_before"], record["status_after"]
    employed, unemployed = ActivityStatus.EMPLOYED, ActivityStatus.UNEMPLOYED
    finders = (before == unemployed) & (after == employed)
    losers = (before == employed) & (after == unemployed)
    return finders, losers, ~(finders | losers)


def test_labour_clearing_moves_people_between_unemployment_and_employment(instrumented_period):
    _, record = instrumented_period
    finders, losers, _ = _status_groups(record)
    assert finders.any() and losers.any()


def test_known_receipts_use_the_unemployment_benefit_paid_this_period(instrumented_period):
    country, record = instrumented_period
    finders, losers, unchanged = _status_groups(record)
    planned, paid = record["planned_ub_individual"], country.individuals.ts.current(UB_NAME)
    # Job finders are paid no benefit, job losers are paid their entitlement, everyone else is unchanged.
    np.testing.assert_array_equal(paid[finders], 0.0)
    assert np.all(paid[losers] > 0.0)
    for group in (finders, losers, unchanged):
        np.testing.assert_array_equal(planned[group], paid[group])


def _assert_planned_benefits_equal_paid(country, record):
    hh = country.households
    pension, unemployment, other = record["planned_components"][:3]
    np.testing.assert_allclose(pension, hh.ts.current("income_public_pension"), rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(unemployment, hh.ts.current("income_unemployment_benefits"), rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(other, hh.ts.current("income_other_social_transfers"), rtol=1e-12, atol=0.0)
    np.testing.assert_array_equal(record["planned_ub_individual"], country.individuals.ts.current(UB_NAME))


def test_planned_household_benefits_equal_paid_household_benefits(instrumented_period):
    _assert_planned_benefits_equal_paid(*instrumented_period)


def test_planned_benefits_equal_paid_benefits_with_minimum_income(instrumented_minimum_income_period):
    country, record = instrumented_minimum_income_period
    hh = country.households
    assert country.household_support.mode == "minimum_income"
    _assert_planned_benefits_equal_paid(country, record)
    # T is planned and paid as the entitlement fixed at the start of the period.
    entitlement = hh.ts.current("minimum_income_entitlement")
    assert np.any(entitlement > 0.0)
    np.testing.assert_array_equal(record["planned_components"][4], entitlement)
    np.testing.assert_array_equal(hh.ts.current("income_minimum_income"), entitlement)


def test_minimum_income_level_uses_the_benefit_price_level(instrumented_minimum_income_period):
    country, _ = instrumented_minimum_income_period
    hh, economy, government = country.households, country.economy, country.central_government.ts
    cpi_series = economy.consumer_price_level_series_name()
    assert country.benefit_price_level() == economy.ts.prev(cpi_series)[0] != economy.current_consumer_price_level()
    unit_level = country._compute_minimum_income_level(
        transfer_index=government.current("total_other_benefits")[0] / government.initial("total_other_benefits")[0],
        cpi_ratio=1.0,
    )
    np.testing.assert_allclose(
        hh.ts.current("minimum_income_level"),
        unit_level * country.benefit_price_level() / economy.initial_consumer_price_level(),
        rtol=1e-12,
    )


def test_benefits_are_priced_at_last_realised_cpi(instrumented_period):
    country, record = instrumented_period
    economy, hh, individuals = country.economy, country.households, country.individuals
    cpi_series = economy.consumer_price_level_series_name()
    cpi_now, cpi_prev = economy.current_consumer_price_level(), economy.ts.prev(cpi_series)[0]
    assert cpi_now != cpi_prev
    assert record["planned_benefit_price_level"] == cpi_prev
    assert country.benefit_price_level() == cpi_prev
    household_id = individuals.states["Corresponding Household ID"]
    np.testing.assert_allclose(
        hh.ts.current("income_unemployment_benefits"),
        cpi_prev * hh.aggregate_individual_amount(individuals.ts.current(UB_NAME), household_id),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        hh.ts.current("income_public_pension"),
        cpi_prev * hh.aggregate_individual_amount(individuals.ts.current("public_pension_benefits"), household_id),
        rtol=1e-12,
    )


def test_fiscal_benefit_totals_equal_household_receipts(instrumented_period):
    country, _ = instrumented_period
    hh, cg = country.households.ts, country.central_government.ts
    assert cg.current("total_unemployment_benefits")[0] == hh.current("income_unemployment_benefits").sum()
    assert cg.current("total_public_pension_benefits")[0] == hh.current("income_public_pension").sum()
    assert cg.current("total_other_social_transfers")[0] == hh.current("income_other_social_transfers").sum()


@pytest.mark.parametrize("fixture_name", ["instrumented_period", "instrumented_minimum_income_period"])
def test_deficit_benefit_outlay_equals_household_social_receipts(fixture_name, request):
    country, _ = request.getfixturevalue(fixture_name)
    cg = country.central_government.ts
    spending = country.government_entities.ts.current("nominal_amount_spent_in_lcu").sum()
    benefit_outlay = (
        cg.current("deficit")[0] - spending - cg.current("interest_payments_on_debt")[0] + cg.current("revenue")[0]
    )
    receipts = country.households.ts.current("income_social_transfers").sum()
    assert benefit_outlay == pytest.approx(receipts, rel=1e-9)
    assert cg.current("debt")[0] - cg.prev("debt")[0] == pytest.approx(cg.current("deficit")[0], rel=1e-9)


@pytest.mark.parametrize("fixture_name", ["instrumented_period", "instrumented_minimum_income_period"])
def test_expected_benefit_terms_equal_paid_benefit_terms(fixture_name, request):
    country, record = request.getfixturevalue(fixture_name)
    hh = country.households
    assert record["expected_inflation"] != 0.0
    # Pass 2 runs after labour clearing: every expected benefit term equals the amount paid.
    for expected, paid in EXPECTED_BENEFITS.items():
        np.testing.assert_allclose(record[2][expected], hh.ts.current(paid), rtol=1e-12, atol=0.0)
    # Pass 1 runs before labour clearing: UB differs only for households with a member whose status changed.
    changed = np.zeros(hh.ts.current("n_households"), dtype=bool)
    moved = record["status_before"] != record["status_after"]
    changed[country.individuals.states["Corresponding Household ID"][moved]] = True
    for expected, paid in EXPECTED_BENEFITS.items():
        keep = ~changed if paid == "income_unemployment_benefits" else slice(None)
        np.testing.assert_allclose(record[1][expected][keep], hh.ts.current(paid)[keep], rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("fixture_name", ["instrumented_period", "instrumented_minimum_income_period"])
def test_expected_period_wage_equals_paid_wage_in_both_passes(fixture_name, request):
    country, record = request.getfixturevalue(fixture_name)
    hh = country.households
    assert record["expected_inflation"] != 0.0
    paid = hh.ts.current("income_employee")
    assert paid.sum() > 0.0
    # Pass 2 runs after labour clearing: the expected wage is the wage paid, with no expected-inflation uplift.
    np.testing.assert_allclose(record[("wage", 2)]["expected"], paid, rtol=1e-12, atol=0.0)
    # Pass 1 runs before clearing: the locked wage price level on the pre-clearing wages, again with no uplift.
    # Contract wages are set after pass 1, so pass 1 can differ from the wage paid, but not by the inflation uplift.
    pass_1 = record[("wage", 1)]
    np.testing.assert_array_equal(pass_1["expected"], pass_1["locked_wages"])


@pytest.mark.parametrize("fixture_name", ["instrumented_period", "instrumented_minimum_income_period"])
def test_known_receipts_and_individual_expected_wage_use_the_paid_wage(fixture_name, request):
    country, record = request.getfixturevalue(fixture_name)
    hh = country.households
    # Known receipts: the wage inside financing income is the wage paid.
    pass_2 = record[("wage", 2)]
    paid_benefits = sum(hh.ts.current(paid) for paid in EXPECTED_BENEFITS.values()) + hh.ts.current(
        "income_minimum_income"
    )
    other_expected = (
        hh.ts.current("expected_income") - pass_2["expected"] - hh.ts.current("expected_income_social_transfers")
    )
    np.testing.assert_allclose(
        hh.ts.current("financing_income"),
        other_expected + hh.ts.current("income_employee") + paid_benefits,
        rtol=1e-12,
        atol=1e-6,
    )
    # Individual expected income of the employed is the locked-price wage (benefits aside: employed get none).
    employed = pass_2["employed"]
    np.testing.assert_allclose(
        pass_2["expected_individual"][employed], pass_2["nominal_individual"][employed], rtol=1e-12, atol=0.0
    )
