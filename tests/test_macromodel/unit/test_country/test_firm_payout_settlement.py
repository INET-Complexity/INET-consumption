"""Exercise the country ordering of actual tax, facility repayment and payouts."""

import numpy as np
import pytest

from macromodel.configurations import CountryConfiguration, SimulationConfiguration
from macromodel.simulation import Simulation


@pytest.mark.parametrize(
    "carried,sales,actual_tax,expected_repayment,expected_paid",
    [
        (100, 40, 0, 70, 0),
        (100, 40, 10, 60, 0),
        (100, 40, 20, 50, 0),
        (100, 40, 30, 40, 0),
        (100, 75, 0, 100, 5),
        (100, 100, 0, 100, 10),
        (0, 40, 0, 0, 10),
    ],
)
def test_actual_tax_cash_services_carried_facility_before_dividends(
    datawrapper, monkeypatch, carried, sales, actual_tax, expected_repayment, expected_paid
):
    simulation = Simulation.from_datawrapper(
        datawrapper=datawrapper,
        simulation_configuration=SimulationConfiguration(
            country_configurations={"FRA": CountryConfiguration()}, seed=0
        ),
    )
    country = simulation.countries["FRA"]
    firms = country.firms
    n_firms = firms.ts.current("n_firms")

    def full(value):
        return np.full(n_firms, value, dtype=float)

    draw = firms.draw_operating_revolving_facility
    opening_lengths = {}

    def draw_with_realised_shortfall(**kwargs):
        # Enter the real country settlement with disappointing sales, no
        # cleared refinancing, and a profit-backed proposal from last period.
        values = {
            "deposits": 0,
            "nominal_amount_sold_in_lcu": sales,
            "total_wage": 80,
            "capital_formation_taxes_paid": 0,
            "dividend_fund_declared_distribution": 10,
            "dividend_fund_planned_distribution": 10,
            "dividend_fund_after_tax_earnings": 100,
            "operating_revolving_opening_balance": carried,
            "operating_revolving_interest_paid": 0,
        }
        for field in (
            "received_short_term_credit",
            "received_ordinary_short_term_credit",
            "received_operating_refinance_credit",
            "received_overdraft_refinance_credit",
            "received_debt_rollover_credit",
            "received_long_term_credit",
            "firm_settlement_opening_interest_arrears",
            "firm_settlement_opening_principal_arrears",
            "firm_settlement_contractual_interest_due",
            "firm_settlement_contractual_principal_due",
            "firm_settlement_scheduled_interest_due",
            "firm_settlement_scheduled_principal_due",
        ):
            values[field] = 0
        for field, value in values.items():
            firms.ts.override_current(field, full(value))
        firms.ts.override_current("nominal_amount_spent_in_lcu", np.zeros((n_firms, firms.n_industries)))
        monkeypatch.setitem(firms.states, "is_insolvent", np.zeros(n_firms, dtype=bool))
        for field in ("operating_revolving_repayment", "operating_revolving_closing_balance"):
            opening_lengths[field] = len(getattr(firms.ts, field))
        return draw(
            intermediate_purchases=full(0),
            taxes_paid_on_production=full(0),
            non_loan_interest_paid=full(0),
            corporate_tax_obligation_preview=kwargs["corporate_tax_obligation_preview"],
        )

    class SettlementRecorded(Exception):
        pass

    def stop_before_balance_sheets():
        # The actual Country.update_realised_metrics caller has now recorded
        # tax, facility and dividend flows; unrelated later stages are outside
        # this deliberately controlled realised-sales scenario.
        raise SettlementRecorded

    monkeypatch.setattr(country, "current_firm_corporate_tax_obligation_preview", lambda: full(20))
    monkeypatch.setattr(firms, "draw_operating_revolving_facility", draw_with_realised_shortfall)
    monkeypatch.setattr(firms, "direct_tfp_investment_cash_expense", lambda: full(0))
    monkeypatch.setattr(firms, "compute_interest_paid_on_deposits", lambda **kwargs: full(0))
    monkeypatch.setattr(firms, "compute_profits", lambda: full(actual_tax / 0.25))
    country.central_government.states["Profit Tax"] = 0.25
    monkeypatch.setitem(
        country.central_government.states,
        "Taxes Less Subsidies Rates",
        np.zeros_like(country.central_government.states["Taxes Less Subsidies Rates"]),
    )
    monkeypatch.setattr(firms, "compute_gross_operating_surplus_mixed_income", stop_before_balance_sheets)

    with pytest.raises(SettlementRecorded):
        simulation.iterate()

    np.testing.assert_allclose(firms.ts.current("corporate_taxes_paid"), actual_tax)
    np.testing.assert_allclose(firms.ts.current("operating_revolving_current_draw"), 110)
    np.testing.assert_allclose(firms.ts.current("operating_revolving_repayment"), expected_repayment)
    np.testing.assert_allclose(
        firms.ts.current("operating_revolving_closing_balance"), carried + 110 - expected_repayment
    )
    np.testing.assert_allclose(firms.ts.current("dividend_fund_settlement_debit"), expected_paid)
    np.testing.assert_allclose(firms.ts.current("dividend_fund_final_declared_distribution"), expected_paid)
    np.testing.assert_allclose(firms.ts.current("dividend_fund_expired_distribution"), 10 - expected_paid)
    for field, length in opening_lengths.items():
        assert len(getattr(firms.ts, field)) == length + 1
