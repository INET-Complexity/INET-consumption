"""Canonical floor units at external-data ingestion and across planning passes."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import macromodel.country.country as country_module
from macromodel.configurations import CountryConfiguration, SimulationConfiguration, load_country_configuration
from macromodel.country import Country
from macromodel.simulation import Simulation


@pytest.mark.parametrize("scale", [1.0, 5000.0])
@pytest.mark.parametrize("months", [1, 3, 12])
def test_insee_floor_has_same_household_purchasing_power_at_each_scale(monkeypatch, scale, months):
    monkeypatch.setattr(country_module, "load_insee_smic_annual_table", lambda path=None: pd.Series({2014: 1128.7}))
    base = Country._select_net_smic_base(
        initial_year=2014, fallback_net_smic=99.0, country_name="FRA", population_scale_factor=scale
    )
    # Nonunit initial CPI ensures population expansion is independent of rebasing.
    floor = Country._compute_subsistence_consumption_from_units(
        net_smic_base=base,
        current_cpi=1.5,
        initial_cpi=1.2,
        consumption_units=np.array([1.0, 1.8]),
        time_unit=months,
    )
    # Physical monthly floor is half SMIC per CU, at the planning price level.
    np.testing.assert_allclose(floor / scale / months, np.array([564.35, 1015.83]) * 1.25)


@pytest.mark.parametrize("reason", ["non_fra", "missing_year", "nan", "load_error"])
def test_model_unit_fallback_is_not_population_expanded_again(monkeypatch, reason):
    def table(path=None):
        if reason == "non_fra":
            raise AssertionError("Non-FRA must not load INSEE")
        if reason == "load_error":
            raise OSError("missing table")
        return pd.Series({2015: 1128.7}) if reason == "missing_year" else pd.Series({2014: np.nan})

    monkeypatch.setattr(country_module, "load_insee_smic_annual_table", table)
    assert (
        Country._select_net_smic_base(
            initial_year=2014,
            fallback_net_smic=495000.0,
            country_name="ROW" if reason == "non_fra" else "FRA",
            population_scale_factor=5000.0,
        )
        == 495000.0
    )


@pytest.mark.parametrize("scale", [0.0, -1.0, np.nan, np.inf])
def test_invalid_population_scale_is_rejected(scale):
    with pytest.raises(ValueError, match="finite and positive"):
        Country._select_net_smic_base(
            initial_year=2014, fallback_net_smic=99.0, country_name="FRA", population_scale_factor=scale
        )


def test_canonical_floor_reaches_both_planning_passes_and_reset(datawrapper, monkeypatch):
    root = Path(__file__).resolve().parents[4]
    production = load_country_configuration(root / "run_model/config/country_config_FRA.yaml", country_iso3="FRA")
    config = CountryConfiguration()
    config.households = production.households
    config.households.functions.wealth.parameters["frm_coefficients_path"] = str(
        root / "run_model/data/raw_data/portfolio/FR_portfolio_frm_coefficients.json"
    )
    config.banks.parameters.household_consumption_loan_maturity = (
        production.banks.parameters.household_consumption_loan_maturity
    )
    config.banks.parameters.requires_household_underwriting_snapshot = True
    config.consumer_credit = production.consumer_credit
    config.credit_market = production.credit_market
    monkeypatch.setattr(country_module, "load_insee_smic_annual_table", lambda path=None: pd.Series({2014: 1128.7}))
    simulation = Simulation.from_datawrapper(
        datawrapper=datawrapper,
        simulation_configuration=SimulationConfiguration(country_configurations={"FRA": config}, seed=15),
    )
    country = simulation.countries["FRA"]
    # Exercise both target calls without running the separate macro estimation phase.
    monkeypatch.setattr(country.economy, "current_expected_consumer_period_inflation", lambda: 0.0)
    hh = country.households
    scale = float(country.scale)
    cu = hh._current_consumption_units()
    initial_floor = 1693.05 * scale * cu
    np.testing.assert_allclose(country.economy.ts.current("subsistence_consumption"), initial_floor)
    np.testing.assert_allclose(country.economy.ts.current("net_smic_base"), [1128.7 * scale])
    assert np.all(hh.ts.current("expected_behavioural_income") >= initial_floor)
    assert np.all(hh.ts.current("income_bridge_current") >= initial_floor)
    captured = []
    original = hh.functions["consumption"].compute_target_consumption

    def capture(**kwargs):
        captured.append(kwargs["subsistence_income"].copy())
        return original(**kwargs)

    monkeypatch.setattr(hh.functions["consumption"], "compute_target_consumption", capture)
    country._set_household_target_demand(replace_current=False)
    length = len(country.economy.ts.historic("subsistence_consumption"))
    country._set_household_target_demand(replace_current=True)
    assert len(country.economy.ts.historic("subsistence_consumption")) == length
    assert len(captured) == 2
    for floor in captured:
        np.testing.assert_allclose(floor, initial_floor)
    np.testing.assert_allclose(country.economy.ts.current("net_smic_base"), [1128.7 * scale])
    np.testing.assert_allclose(country.economy.ts.current("subsistence_consumption"), initial_floor)
    assert np.all(hh.ts.current("expected_behavioural_income") >= initial_floor)
    assert np.all(hh.ts.current("income_bridge_current") >= initial_floor)

    country.reset(config)
    assert len(country.economy.ts.historic("subsistence_consumption")) == 1
    np.testing.assert_allclose(country.economy.ts.current("subsistence_consumption"), initial_floor)
    country._update_subsistence_consumption(replace_current=False)
    country._update_subsistence_consumption(replace_current=True)
    np.testing.assert_allclose(country.economy.ts.current("subsistence_consumption"), initial_floor)
    np.testing.assert_allclose(country.economy.ts.current("net_smic_base"), [1128.7 * scale])


# --- Increment (c): floor parameter, housing deduction and consumption-unit weights from configuration ---

_PRODUCTION_CONFIG = Path(__file__).resolve().parents[4] / "run_model/config/country_config_FRA.yaml"


def test_production_reference_resolves_the_paper_block():
    production = load_country_configuration(_PRODUCTION_CONFIG, country_iso3="FRA")
    parameters = country_module._read_subsistence_floor_parameters(production.subsistence_consumption)
    assert parameters.smic_share == 0.5
    assert parameters.cu_weights == (1.0, 0.5, 0.3)
    assert parameters.housing_deduction_monthly == (59.92, 119.84, 148.30)


@pytest.mark.parametrize("block", [None, {}])
def test_empty_block_reproduces_the_previous_floor_exactly(block):
    parameters = country_module._read_subsistence_floor_parameters(block)
    assert parameters == country_module.SubsistenceFloorParameters(0.5, (1.0, 0.5, 0.3), (0.0, 0.0, 0.0))
    assert CountryConfiguration().subsistence_consumption == {}
    units = np.array([1.0, 1.3, 1.8, 2.1, 3.4])
    base, cpi, cpi0, months = 1128.7 * 5000.0, 1.0731, 0.9917, 3
    floor = Country._compute_subsistence_consumption_from_units(
        net_smic_base=base,
        current_cpi=cpi,
        initial_cpi=cpi0,
        consumption_units=units,
        time_unit=months,
        smic_share=parameters.smic_share,
        housing_deduction=parameters.housing_deduction_by_household(
            np.array([1, 2, 2, 3, 6]), population_scale_factor=5000.0
        ),
    )
    # The expression used before the block was wired; equality is exact, not approximate.
    np.testing.assert_array_equal(floor, 0.5 * (base * max(cpi / cpi0, 0.0) * float(months)) * units)


@pytest.mark.parametrize(
    "block",
    [
        {"smic_share": 0.0},
        {"smic_share": 1.01},
        {"smic_share": np.nan},
        {"smic_share": "0.5"},
        {"smic_share": True},
        {"cu_weights": {"first_adult": 1.0, "additional_adult": 0.5}},
        {"cu_weights": {"first_adult": 1.0, "additional_adult": 0.0, "child": 0.3}},
        {"cu_weights": {"first_adult": 1.0, "additional_adult": np.inf, "child": 0.3}},
        {"housing_deduction_monthly": {1: 59.92, 2: 119.84}},
        {"housing_deduction_monthly": {1: 59.92, 2: 119.84, 3: 148.3, 4: 150.0}},
        {"housing_deduction_monthly": {1: -1.0, 2: 119.84, 3: 148.3}},
        {"housing_deduction_monthly": {1: None, 2: 119.84, 3: 148.3}},
        {"smic_monthly_override": None},
    ],
)
def test_invalid_floor_block_is_rejected(block):
    with pytest.raises(ValueError, match="subsistence_consumption"):
        country_module._read_subsistence_floor_parameters(block)


def test_housing_deduction_keys_may_be_strings():
    parameters = country_module._read_subsistence_floor_parameters(
        {"housing_deduction_monthly": {"1": 10.0, "2": 20.0, "3": 30.0}}
    )
    assert parameters.housing_deduction_monthly == (10.0, 20.0, 30.0)


def test_housing_deduction_by_household_size_and_population_scale():
    parameters = country_module.SubsistenceFloorParameters(housing_deduction_monthly=(59.92, 119.84, 148.30))
    sizes = np.array([0, 1, 2, 3, 5])
    np.testing.assert_allclose(
        parameters.housing_deduction_by_household(sizes, population_scale_factor=1.0),
        [59.92, 59.92, 119.84, 148.30, 148.30],
    )
    np.testing.assert_allclose(
        parameters.housing_deduction_by_household(sizes, population_scale_factor=5000.0),
        np.array([59.92, 59.92, 119.84, 148.30, 148.30]) * 5000.0,
    )


@pytest.mark.parametrize("scale", [1.0, 5000.0])
@pytest.mark.parametrize("months", [1, 3])
def test_floor_subtracts_the_deduction_at_the_planning_price_level(scale, months):
    parameters = country_module.SubsistenceFloorParameters(housing_deduction_monthly=(59.92, 119.84, 148.30))
    floor = Country._compute_subsistence_consumption_from_units(
        net_smic_base=1128.7 * scale,
        current_cpi=1.5,
        initial_cpi=1.2,
        consumption_units=np.array([1.0, 1.5, 2.1]),
        time_unit=months,
        housing_deduction=parameters.housing_deduction_by_household(np.array([1, 2, 4]), population_scale_factor=scale),
    )
    # Ordinary euros per month at base prices: 0.5 x 1128.70 x CU - H.
    monthly = np.array([564.35 - 59.92, 846.525 - 119.84, 1185.135 - 148.30])
    np.testing.assert_allclose(floor / scale / months, monthly * 1.25)


def test_floor_is_linear_in_smic_share_and_never_negative():
    arguments = dict(net_smic_base=1000.0, current_cpi=1.0, initial_cpi=1.0, time_unit=3)
    units = np.array([1.0, 1.8])
    half = Country._compute_subsistence_consumption_from_units(consumption_units=units, smic_share=0.5, **arguments)
    quarter = Country._compute_subsistence_consumption_from_units(consumption_units=units, smic_share=0.25, **arguments)
    np.testing.assert_allclose(quarter, 0.5 * half)
    floored = Country._compute_subsistence_consumption_from_units(
        consumption_units=units, smic_share=0.5, housing_deduction=np.array([600.0, 100.0]), **arguments
    )
    np.testing.assert_allclose(floored, [0.0, 3 * (900.0 - 100.0)])


def test_configured_floor_block_reaches_households_and_country(datawrapper, monkeypatch):
    config = CountryConfiguration()
    config.subsistence_consumption = {
        "smic_share": 0.4,
        "cu_weights": {"first_adult": 1.0, "additional_adult": 0.6, "child": 0.25},
        "housing_deduction_monthly": {1: 59.92, 2: 119.84, 3: 148.30},
    }
    monkeypatch.setattr(country_module, "load_insee_smic_annual_table", lambda path=None: pd.Series({2014: 1128.7}))
    simulation = Simulation.from_datawrapper(
        datawrapper=datawrapper,
        simulation_configuration=SimulationConfiguration(country_configurations={"FRA": config}, seed=15),
    )
    country = simulation.countries["FRA"]
    hh = country.households
    scale = float(country.scale)
    assert hh.cu_weights == (1.0, 0.6, 0.25)
    older, younger = np.array(hh._consumption_unit_composition_signature).T
    np.testing.assert_allclose(hh.current_consumption_units(), 1.0 + 0.6 * np.maximum(older - 1, 0) + 0.25 * younger)
    np.testing.assert_array_equal(hh.current_household_sizes(), older + younger)
    deduction = np.array([59.92, 119.84, 148.30])[np.clip(older + younger, 1, 3) - 1]
    expected = 3 * np.maximum(0.4 * 1128.7 * hh.current_consumption_units() - deduction, 0.0) * scale
    np.testing.assert_allclose(country.economy.ts.current("subsistence_consumption"), expected)
    country._update_subsistence_consumption(replace_current=True)
    np.testing.assert_allclose(country.economy.ts.current("subsistence_consumption"), expected)
