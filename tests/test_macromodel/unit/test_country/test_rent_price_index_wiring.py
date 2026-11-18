"""Rent price index wiring on a real Country (issue #197)."""

from pathlib import Path

import numpy as np
import pandas as pd

import macromodel.country.country as country_module
from macromodel.configurations import CountryConfiguration, SimulationConfiguration, load_country_configuration
from macromodel.simulation import Simulation


def build_country(datawrapper, monkeypatch):
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
    return simulation.countries["FRA"], config


def test_floor_groups_and_rent_base_are_wired(datawrapper, monkeypatch):
    country, config = build_country(datawrapper, monkeypatch)
    properties = country.housing_market.states["properties"]

    # The unset runtime floor takes the data value.
    data_floor = country.data_minimum_annual_rental_yield
    assert country.households.functions["property"].minimum_annual_rental_yield == data_floor

    # The index base is the dwelling rents at t=0, in property order.
    np.testing.assert_allclose(country._previous_property_rents, properties["Rent"].to_numpy())

    # The dwelling groups reproduce the two rent flows that enter GDP.
    rent, imputed = country.households.compute_rent(
        housing_data=properties,
        social_housing_function=country.central_government.functions["social_housing"],
        current_unemployment_benefits_by_individual=1.0,
    )
    tenure = country.households.states["Dwelling Tenure"]
    social = country.households.states["Rent paid to Government"]
    np.testing.assert_allclose(properties["Rent"].to_numpy()[tenure == 3].sum(), rent.sum() - social)
    np.testing.assert_allclose(properties["Rent"].to_numpy()[np.isin(tenure, [1, 2, 4])].sum(), imputed.sum())

    # A reset with a scenario floor uses it and restores the index base.
    properties["Rent"] *= 1.5
    config.households.functions.property.parameters["minimum_annual_rental_yield"] = 0.03
    country.reset(configuration=config)
    assert country.households.functions["property"].minimum_annual_rental_yield == 0.03
    np.testing.assert_allclose(
        country._previous_property_rents, country.housing_market.states["properties"]["Rent"].to_numpy()
    )
