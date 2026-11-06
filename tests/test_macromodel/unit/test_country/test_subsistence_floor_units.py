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
