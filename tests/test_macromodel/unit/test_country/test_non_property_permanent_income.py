"""PI coverage is wages/transfers; cash and consumption-income coverage differ."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from macromodel.country import Country
from macromodel.timeseries import TimeSeries


@pytest.mark.parametrize("scale", [1.0, 1e6])
def test_initial_pi_excludes_property_receipts_and_preserves_behavioural_income(scale):
    n = np.array([100.0, 0.0, -20.0]) * scale
    zero = np.zeros(3)
    country = Country.__new__(Country)
    country._staged_household_income_shock = None
    country.households = SimpleNamespace(
        ts=TimeSeries(
            expected_income_employee=n,
            expected_income_social_transfers=zero.copy(),
            expected_income_financial_assets=np.full(3, 30.0 * scale),
            income_rental=np.full(3, 20.0 * scale),
            **{
                name: zero.copy()
                for name in (
                    "income_bridge_current",
                    "income_bridge_property",
                    "income_bridge_floor_used",
                    "income_bridge_component_floor_used",
                )
            },
        ),
        functions={"wealth": object()},
        compute_interest_paid_on_deposits=lambda **kwargs: np.full(3, -10.0 * scale),
        current_consumption_imputed_rent=lambda: np.full(3, 5.0 * scale),
    )
    country.banks = SimpleNamespace(
        ts=TimeSeries(
            interest_rate_on_household_deposits=np.zeros(1),
            overdraft_rate_on_household_deposits=np.zeros(1),
        )
    )
    country.credit_market = SimpleNamespace(preview_opening_household_interest=lambda: np.full(3, 2.0 * scale))
    country.economy = SimpleNamespace(ts=TimeSeries(subsistence_consumption=np.full(3, scale)))

    country._record_behavioural_income(expected=True, initial=True)

    ts = country.households.ts
    np.testing.assert_allclose(ts.current("income_bridge_current"), np.array([100.0, 1.0, 1.0]) * scale)
    np.testing.assert_array_equal(ts.current("income_bridge_property"), zero)
    np.testing.assert_array_equal(ts.current("income_bridge_floor_used"), [0.0, 1.0, 1.0])
    np.testing.assert_array_equal(ts.current("income_bridge_component_floor_used"), [0.0, 0.0, 1.0])
    np.testing.assert_allclose(
        ts.current("expected_behavioural_income"), np.array([np.sqrt(100.0 * 160.0) + 3.0, 3.0, 3.0]) * scale
    )
    np.testing.assert_allclose(ts.current("expected_behavioural_income_deposit_receipts"), 10.0 * scale)
    np.testing.assert_allclose(ts.current("expected_behavioural_income_financial"), 30.0 * scale)
    np.testing.assert_allclose(ts.current("expected_behavioural_income_rental"), 20.0 * scale)


def test_common_forecast_income_excludes_property_and_responds_to_wages_and_transfers():
    country = Country.__new__(Country)
    country.start_period = pd.Period("2014Q1", freq="Q")
    country.households = SimpleNamespace(
        ts=TimeSeries(income_employee=np.array([80.0, 20.0]), income_social_transfers=np.array([10.0, 10.0]))
    )
    country.households.ts.income_employee.append(np.array([100.0, 50.0]))
    country.households.ts.income_social_transfers.append(np.array([20.0, 10.0]))
    country.households.ts["income"] = np.array([500.0, 500.0])
    country.households.ts.income.append(np.array([1000.0, 1000.0]))
    country.economy = SimpleNamespace(ts=TimeSeries(cpi_fixed_basket=[1.0], unemployment_rate=[0.1]))
    country.economy.ts.cpi_fixed_basket.append([2.0])
    country.economy.ts.unemployment_rate.append([0.2])
    country.individuals = SimpleNamespace(ts=TimeSeries(n_individuals=[10]))
    country.individuals.ts.n_individuals.append([20])
    country.central_bank = SimpleNamespace(ts=TimeSeries(policy_rate=[0.01]))
    country.central_bank.ts.policy_rate.extend([[0.02], [0.03]])
    before = deepcopy(country.households.ts.dicts)

    sources = country._permanent_income_simulation_sources()

    # Wages/transfers total 120 then 180; real per-capita income is 12 then 4.5.
    np.testing.assert_allclose(sources.real_pc_income, [100.0, 37.5])
    np.testing.assert_allclose(sources.policy_rate, [0.01, 0.02])
    for key, history in before.items():
        np.testing.assert_array_equal(country.households.ts.dicts[key], history)

    country.households.ts.override_current("income", np.array([1e9, 1e9]))
    np.testing.assert_array_equal(country._permanent_income_simulation_sources().real_pc_income, sources.real_pc_income)
    country.households.ts.override_current("income_employee", np.array([160.0, 50.0]))
    np.testing.assert_allclose(country._permanent_income_simulation_sources().real_pc_income, [100.0, 50.0])
    country.households.ts.override_current("income_social_transfers", np.array([50.0, 40.0]))
    np.testing.assert_allclose(country._permanent_income_simulation_sources().real_pc_income, [100.0, 62.5])

    country.households.ts.income_social_transfers.pop()
    with pytest.raises(ValueError, match="zip"):
        country._permanent_income_simulation_sources()
