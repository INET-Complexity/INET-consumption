"""Minimum income entitlement orchestration and booking in Country (PLAN.md section 5.3 steps 1-2 and 5).

Invariants (PLAN.md section 6): 1 (sum of T = government total, in the deficit once), 2 (social income
components), 4 (T computed and booked once per period), 8 (deposits credited once), 9 (MPC receipt absent
from Y), plus T identical across the two planning passes and independent of period-t outcomes.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from macromodel.configurations import CountryConfiguration, SimulationConfiguration
from macromodel.configurations.country_config_loader import load_country_configuration
from macromodel.country.country import _read_household_support_parameters
from macromodel.simulation import Simulation

RSA = 499.31
COMPONENTS = (
    "income_employee",
    "income_unemployment_benefits",
    "income_public_pension",
    "income_other_social_transfers",
    "income_rental",
)
ENTITLEMENT_SERIES = ("minimum_income_level", "minimum_income_assessed", "minimum_income_entitlement")


def _use_minimum_income(country, **minimum_income):
    block = {"mode": "minimum_income"}
    if minimum_income:
        block["minimum_income"] = minimum_income
    country.household_support = _read_household_support_parameters(block)


def _fix_benefit_setter(country, monkeypatch, index=1.5):
    """Replace the data-driven setter (NaN on the unit fixture) by a known real level ratio I_t."""
    government = country.central_government

    def update_benefits(**_kwargs):
        government.ts.total_other_benefits.append([index * government.ts.initial("total_other_benefits")[0]])

    monkeypatch.setattr(government, "update_benefits", update_benefits)


def _set_t_minus_1_rows(country, rng_seed=0):
    """Overwrite the realised t-1 rows with known component values (in model units)."""
    hh = country.households
    n = hh.ts.current("n_households")
    rng = np.random.default_rng(rng_seed)
    rows = {name: rng.uniform(0.0, 2000.0, n) for name in COMPONENTS}
    rows["income_rental"] = rng.uniform(-300.0, 300.0, n)  # net rental income can be negative
    rows["liquid_financial_assets"] = rng.uniform(-5000.0, 20000.0, n)  # overdrafts included
    rows["illiquid_financial_assets"] = rng.uniform(0.0, 50000.0, n)
    for name, values in rows.items():
        hh.ts.override_current(name, values)
    return rows


def _expected_entitlement(country, rows, *, index, cpi_ratio):
    time_unit = country.economy.time_unit
    level = RSA * country.scale * time_unit * country.households._current_consumption_units() * index * cpi_ratio
    assets = np.maximum(rows["liquid_financial_assets"], 0.0) + rows["illiquid_financial_assets"]
    assessed = sum(rows[name] for name in COMPONENTS) + 0.03 * time_unit / 12.0 * assets
    return level, assessed, np.maximum(level - assessed, 0.0)


def _start_period(country):
    """Period-start sequence of update_pre_credit_household_finance_metrics up to the expectations pass."""
    country._update_benefit_planning_metrics()
    country._set_minimum_income_entitlement()


def test_entitlement_assesses_t_minus_1_components_and_gross_financial_assets(test_country, monkeypatch):
    _use_minimum_income(test_country)
    _fix_benefit_setter(test_country, monkeypatch, index=1.5)
    test_country.scale = 2
    initial_cpi = test_country.economy.initial_consumer_price_level()
    monkeypatch.setattr(test_country.economy, "current_consumer_price_level", lambda: 1.2 * initial_cpi)
    rows = _set_t_minus_1_rows(test_country)

    _start_period(test_country)

    level, assessed, entitlement = _expected_entitlement(test_country, rows, index=1.5, cpi_ratio=1.2)
    hh = test_country.households
    np.testing.assert_allclose(hh.ts.current("minimum_income_level"), level, rtol=1e-12)
    np.testing.assert_allclose(hh.ts.current("minimum_income_assessed"), assessed, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(hh.ts.current("minimum_income_entitlement"), entitlement, rtol=1e-12, atol=1e-9)
    recipients = hh.ts.current("minimum_income_entitlement") > 0.0
    assert recipients.any() and not recipients.all()


def test_legacy_mode_stores_zero_entitlement_and_books_zero(test_country, monkeypatch):
    _fix_benefit_setter(test_country, monkeypatch)
    _set_t_minus_1_rows(test_country)

    _start_period(test_country)
    test_country.settle_household_social_transfers()

    for name in (*ENTITLEMENT_SERIES, "income_minimum_income"):
        assert len(test_country.households.ts.historic(name)) == 2
        np.testing.assert_array_equal(test_country.households.ts.current(name), 0.0)


def test_entitlement_is_computed_once_and_booked_once_per_period(test_country, monkeypatch):
    _use_minimum_income(test_country)
    _fix_benefit_setter(test_country, monkeypatch)
    _set_t_minus_1_rows(test_country)

    _start_period(test_country)
    with pytest.raises(RuntimeError, match="computed once per period"):
        test_country._set_minimum_income_entitlement()

    test_country.settle_household_social_transfers()
    with pytest.raises(RuntimeError, match="booked once per period"):
        test_country.settle_household_social_transfers()

    # The next period assesses again.
    test_country._set_minimum_income_entitlement()
    assert len(test_country.households.ts.historic("minimum_income_entitlement")) == 3


def test_booked_top_up_is_a_named_fifth_social_income_component(test_country, monkeypatch):
    _use_minimum_income(test_country)
    _fix_benefit_setter(test_country, monkeypatch)
    test_country.scale = 10_000
    _start_period(test_country)
    stage5 = np.full(test_country.households.ts.current("n_households"), 3.0)
    test_country.set_stage5_subsistence_support_by_household(stage5)
    entitlement = test_country.households.ts.current("minimum_income_entitlement").copy()
    assert (entitlement > 0.0).any()

    components = test_country.compute_realised_household_social_income_components()
    test_country.settle_household_social_transfers()

    hh = test_country.households
    assert len(components) == 5
    np.testing.assert_array_equal(components[3], stage5)
    np.testing.assert_array_equal(components[4], entitlement)
    np.testing.assert_array_equal(hh.ts.current("income_minimum_income"), entitlement)
    np.testing.assert_array_equal(hh.ts.current("stage5_subsistence_support"), stage5)
    np.testing.assert_allclose(
        hh.ts.current("income_social_transfers"),
        hh.ts.current("income_public_pension")
        + hh.ts.current("income_unemployment_benefits")
        + hh.ts.current("income_other_social_transfers")
        + stage5
        + entitlement,
        rtol=1e-14,
    )
    np.testing.assert_allclose(
        test_country.compute_realised_household_social_transfers(), hh.ts.current("income_social_transfers")
    )


def test_planning_passes_add_the_stored_top_up_once(test_country, monkeypatch):
    test_country.scale = 10_000
    legacy = deepcopy(test_country)
    _use_minimum_income(test_country)
    entitlement = None
    for country in (legacy, test_country):
        _fix_benefit_setter(country, monkeypatch)
        _start_period(country)
        entitlement = country.households.ts.current("minimum_income_entitlement")
    assert (entitlement > 0.0).any()

    def planning_rows(country):
        hh = country.households
        return {
            name: hh.ts.current(name).copy()
            for name in ("expected_income_social_transfers", "expected_income", "financing_income")
        }

    for country in (legacy, test_country):
        country._set_household_income_expectations(replace_current=False)
    pass_1 = {name: planning_rows(test_country)[name] - planning_rows(legacy)[name] for name in planning_rows(legacy)}
    for country in (legacy, test_country):
        country._set_household_income_expectations(replace_current=True)
        country.refresh_known_household_receipts()
    pass_2 = {name: planning_rows(test_country)[name] - planning_rows(legacy)[name] for name in planning_rows(legacy)}

    for difference in (*pass_1.values(), *pass_2.values()):
        np.testing.assert_allclose(difference, entitlement, rtol=1e-9, atol=1e-6)
    # Pass 2 replaces: one planning row per period, and T itself is unchanged by the passes.
    assert len(test_country.households.ts.historic("expected_income_social_transfers")) == 2
    assert len(test_country.households.ts.historic("minimum_income_entitlement")) == 2
    np.testing.assert_array_equal(test_country.households.ts.current("minimum_income_entitlement"), entitlement)


def test_top_up_ignores_period_t_planned_rent_and_the_mpc_receipt(test_country, monkeypatch):
    _use_minimum_income(test_country)
    test_country.scale = 10_000
    rows = _set_t_minus_1_rows(test_country)
    twin = deepcopy(test_country)
    n = test_country.households.ts.current("n_households")
    test_country.stage_household_income_shock(np.full(n, 5000.0), exempt_from_support_test=True)
    planned_rent = np.full(n, 1e6)
    monkeypatch.setattr(test_country.households, "compute_rental_income", lambda **_kwargs: planned_rent)

    for country in (twin, test_country):
        _fix_benefit_setter(country, monkeypatch)
        _start_period(country)
    test_country._set_household_income_expectations(replace_current=False)
    test_country._set_household_income_expectations(replace_current=True)

    hh = test_country.households
    np.testing.assert_array_equal(hh.ts.current("income_rental"), planned_rent)
    for name in ENTITLEMENT_SERIES:
        np.testing.assert_array_equal(hh.ts.current(name), twin.households.ts.current(name))
    _, assessed, _ = _expected_entitlement(test_country, rows, index=1.5, cpi_ratio=1.0)
    np.testing.assert_allclose(hh.ts.current("minimum_income_assessed"), assessed, rtol=1e-12, atol=1e-9)


def test_monthly_assessment_reads_the_t_minus_4_to_t_minus_2_history(test_country, monkeypatch):
    _use_minimum_income(test_country)
    _fix_benefit_setter(test_country, monkeypatch, index=1.0)
    monkeypatch.setattr(test_country.economy, "time_unit", 1)
    hh = test_country.households
    n = hh.ts.current("n_households")
    names = (*COMPONENTS, "liquid_financial_assets", "illiquid_financial_assets")
    # Rows 0..4 (row 4 = t-1): row k holds 100 * (k + 1) in every component and stock.
    for name in names:
        hh.ts.override_current(name, np.full(n, 100.0))
        for k in range(1, 5):
            hh.ts.dicts[name].append(np.full(n, 100.0 * (k + 1)))

    _start_period(test_country)

    flows = 5 * np.mean([200.0, 300.0, 400.0])  # rows 1..3 = t-4..t-2
    assets = 2 * 400.0  # closing t-2 stocks (row 3)
    np.testing.assert_allclose(hh.ts.current("minimum_income_assessed"), flows + 0.03 / 12.0 * assets)
    np.testing.assert_allclose(hh.ts.current("minimum_income_level"), RSA * hh._current_consumption_units())


def _production_simulation(datawrapper, household_support):
    production = load_country_configuration(
        Path(__file__).resolve().parents[4] / "run_model/config/country_config_FRA.yaml", country_iso3="FRA"
    )
    config = CountryConfiguration()
    config.households = production.households
    config.households.functions.wealth.parameters["frm_coefficients_path"] = str(
        Path(__file__).resolve().parents[4] / "run_model/data/raw_data/portfolio/FR_portfolio_frm_coefficients.json"
    )
    config.firms.functions.wage_setter = production.firms.functions.wage_setter
    config.banks.parameters.household_consumption_loan_maturity = (
        production.banks.parameters.household_consumption_loan_maturity
    )
    config.banks.parameters.requires_household_underwriting_snapshot = True
    config.consumer_credit = production.consumer_credit
    config.credit_market = production.credit_market
    config.household_support = household_support
    return Simulation.from_datawrapper(
        datawrapper=datawrapper,
        simulation_configuration=SimulationConfiguration(country_configurations={"FRA": config}, seed=15),
    )


def test_simulation_books_top_up_in_social_income_and_deficit_once(datawrapper):
    simulation = _production_simulation(datawrapper, {"mode": "minimum_income"})
    country = simulation.countries["FRA"]
    hh, government = country.households, country.central_government
    assert country.configuration.households.parameters.uses_feasibility_resolver
    paid = 0.0
    for period in range(3):
        simulation.iterate(t=period)
        entitlement = hh.ts.current("minimum_income_entitlement")
        paid += entitlement.sum()
        # Invariant 4 and T fixed for the period: booked T is the entitlement assessed at the start.
        np.testing.assert_array_equal(hh.ts.current("income_minimum_income"), entitlement)
        # Invariant 1: household total equals the government total.
        assert government.ts.current("total_minimum_income")[0] == entitlement.sum()
        # Invariant 2: income_social_transfers = pension + UB + other + Stage 5 + T.
        np.testing.assert_allclose(
            hh.ts.current("income_social_transfers"),
            hh.ts.current("income_public_pension")
            + hh.ts.current("income_unemployment_benefits")
            + hh.ts.current("income_other_social_transfers")
            + hh.ts.current("stage5_subsistence_support")
            + entitlement,
            rtol=1e-12,
            atol=1e-6,
        )
        assert government.ts.current("total_household_social_transfers")[0] == pytest.approx(
            hh.ts.current("income_social_transfers").sum(), rel=1e-12
        )
        # T enters the deficit once.
        expected_deficit = (
            government.ts.current("total_unemployment_benefits")[0]
            + government.ts.current("total_public_pension_benefits")[0]
            + government.ts.current("total_other_social_transfers")[0]
            + government.ts.current("total_necessity_support")[0]
            + government.ts.current("total_minimum_income")[0]
            + country.government_entities.ts.current("nominal_amount_spent_in_lcu").sum()
            + government.ts.current("interest_payments_on_debt")[0]
            - government.ts.current("revenue")[0]
        )
        assert government.ts.current("deficit")[0] == pytest.approx(expected_deficit, rel=1e-12)
        # Invariant 8: T reaches deposits once, through income -> update_wealth (closing cash ledger reconciles).
        assert np.max(np.abs(hh.ts.current("stage5_cash_ledger_residual"))) <= 1e-7
        # Pass-2 expected social income carries T once.
        np.testing.assert_allclose(
            hh.ts.current("expected_income_social_transfers")
            - hh.ts.current("expected_income_public_pension")
            - hh.ts.current("expected_income_unemployment_benefits")
            - hh.ts.current("expected_income_other_social_transfers"),
            entitlement,
            rtol=1e-9,
            atol=1e-5,
        )
    assert paid > 0.0
    simulation.reset()
    assert country._minimum_income_status is None
    assert len(hh.ts.historic("minimum_income_entitlement")) == 1
