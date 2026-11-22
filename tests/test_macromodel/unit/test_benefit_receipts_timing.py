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

UB_NAME = "income_from_unemployment_benefits"


@pytest.fixture(scope="module")
def instrumented_period(datawrapper):
    """Run four periods and record the last period's benefit vectors used for planning and for payment.

    In this fixture the first period with both job finders and job losers is the fourth.
    """
    simulation = Simulation.from_datawrapper(
        datawrapper=datawrapper,
        simulation_configuration=SimulationConfiguration(
            country_configurations={"FRA": CountryConfiguration()}, seed=15
        ),
    )
    country = simulation.countries["FRA"]
    record = {}
    clear, refresh, components = (
        country.clear_labour_market,
        country.refresh_known_household_receipts,
        country.compute_realised_household_social_income_components,
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

    country.clear_labour_market = clear_labour_market
    country.refresh_known_household_receipts = refresh_known_household_receipts
    country.compute_realised_household_social_income_components = compute_components
    for period in range(4):
        simulation.iterate(t=period)
    return country, record


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


def test_planned_household_benefits_equal_paid_household_benefits(instrumented_period):
    country, record = instrumented_period
    hh = country.households
    pension, unemployment, other, _ = record["planned_components"]
    np.testing.assert_allclose(pension, hh.ts.current("income_public_pension"), rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(unemployment, hh.ts.current("income_unemployment_benefits"), rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(other, hh.ts.current("income_other_social_transfers"), rtol=1e-12, atol=0.0)


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
    assert cg.current("total_household_social_transfers")[0] == pytest.approx(
        cg.current("total_unemployment_benefits")[0]
        + cg.current("total_public_pension_benefits")[0]
        + cg.current("total_other_social_transfers")[0]
        + cg.current("total_necessity_support")[0],
        rel=1e-12,
    )
