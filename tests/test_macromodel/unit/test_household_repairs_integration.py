"""Exercise the repaired household accounting through the production FRA rules."""

from pathlib import Path

import numpy as np

from macromodel.configurations import CountryConfiguration, SimulationConfiguration
from macromodel.configurations.country_config_loader import load_country_configuration
from macromodel.simulation import Simulation


def test_production_household_receipts_credit_relief_and_support_reconcile(datawrapper):
    production = load_country_configuration(
        Path(__file__).resolve().parents[3] / "run_model/config/country_config_FRA.yaml", country_iso3="FRA"
    )
    # Retain fixture capital-accounting conventions; use the production
    # household, wage, consumer-credit and clearing specifications together.
    config = CountryConfiguration()
    config.households = production.households
    config.households.functions.wealth.parameters["frm_coefficients_path"] = str(
        Path(__file__).resolve().parents[3] / "run_model/data/raw_data/portfolio/FR_portfolio_frm_coefficients.json"
    )
    config.firms.functions.wage_setter = production.firms.functions.wage_setter
    config.banks.parameters.household_consumption_loan_maturity = (
        production.banks.parameters.household_consumption_loan_maturity
    )
    config.banks.parameters.requires_household_underwriting_snapshot = True
    config.consumer_credit = production.consumer_credit
    config.credit_market = production.credit_market
    simulation = Simulation.from_datawrapper(
        datawrapper=datawrapper,
        simulation_configuration=SimulationConfiguration(country_configurations={"FRA": config}, seed=15),
    )
    for period in range(2):
        simulation.iterate(t=period)
        country = simulation.countries["FRA"]
        hh = country.households
        market = country.credit_market
        np.testing.assert_allclose(hh.ts.current("income_employee").sum(), country.nominal_employee_income().sum())
        np.testing.assert_allclose(
            hh.ts.current("mortgage_debt").sum(), market.compute_outstanding_mortgages_by_bank().sum()
        )
        # Expanded synthetic household cash flows reach billions of LCU.
        # Permit only accumulation roundoff; no balance is clipped in code.
        cash_scale = np.maximum(
            1.0, np.abs(hh.ts.current("income")) + hh.ts.current("nominal_amount_spent_in_lcu").sum(axis=1)
        )
        assert np.all(hh.ts.current("liquid_financial_assets") >= -16 * np.finfo(float).eps * cash_scale)
        assert (
            country.current_stage5_subsistence_support_fiscal_increment()
            == hh.ts.current("stage5_subsistence_support").sum()
        )
        assert np.max(np.abs(hh.ts.current("stage5_cash_ledger_residual"))) <= 1e-7
