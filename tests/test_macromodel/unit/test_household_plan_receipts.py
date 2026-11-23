"""Households plan on the receipts and obligations already known at planning.

Runs a short Simulation in legacy_stage5 and minimum_income modes, with a few
households pushed into overdraft, and checks the plan against the settled
period: dividends are planned on last period's declarations net of withholding.
"""

import numpy as np
import pytest

from tests.test_macromodel.unit.test_country.test_minimum_income_entitlement import _production_simulation

PERIODS = 4
OVERDRAWN_FROM = 2


def _run(simulation):
    """Run, push every 40th household into overdraft at the start of period 2, record each period."""
    country = simulation.countries["FRA"]
    hh = country.households
    records = []
    overdrawn = np.zeros(hh.ts.current("n_households"), dtype=bool)
    for period in range(PERIODS):
        if period == OVERDRAWN_FROM:
            overdrawn[::40] = True
            lfa = hh.ts.current("liquid_financial_assets")
            lfa[overdrawn] = -0.2 * np.abs(hh.ts.current("income")[overdrawn]) - 1.0
        simulation.iterate(t=period)
        records.append(
            {
                "expected_dividends": hh.ts.current("expected_income_dividend_distributions").copy(),
                "dividends": hh.ts.current("income_dividend_distributions").copy(),
                "dividend_forecast_error": hh.ts.current("dividend_income_forecast_error").copy(),
                "firm_expired_distribution": country.firms.ts.current("dividend_fund_expired_distribution").copy(),
            }
        )
    return country, overdrawn, records


@pytest.fixture(scope="module", params=["legacy_stage5", "minimum_income"])
def run(request, datawrapper):
    return _run(_production_simulation(datawrapper, {"mode": request.param}))


def test_planned_dividends_equal_settled_dividends_when_no_cap_binds(run):
    _, _, records = run
    assert sum(record["dividends"].sum() for record in records) > 0.0
    for record in records:
        np.testing.assert_allclose(
            record["dividend_forecast_error"], record["dividends"] - record["expected_dividends"], rtol=0.0, atol=0.0
        )
        if record["firm_expired_distribution"].sum() == 0.0:
            np.testing.assert_allclose(record["expected_dividends"], record["dividends"], rtol=1e-12, atol=1e-9)
        else:
            # A binding payer cap leaves part of a declaration unpaid: only a negative error is possible.
            assert np.all(record["dividend_forecast_error"] <= 1e-9)
