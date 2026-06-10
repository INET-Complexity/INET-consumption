from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from macromodel.utils.prehooks.irf_shocks import (
    ShockSpec,
    create_government_consumption_shock_hook,
    create_policy_rate_shock_hook,
    create_tax_rate_shock_hook,
)


class DummyPolicyRate:
    def compute_rate(self, *, shock=0.0, **_kwargs):
        return 0.02 + shock


def test_policy_rate_shock_adds_rate_points_only_during_active_period():
    spec = ShockSpec(name="rate", kind="policy_rate", period=1, magnitude=0.01)
    central_bank = SimpleNamespace(functions={"policy_rate": DummyPolicyRate()})
    simulation = SimpleNamespace(countries={"FRA": SimpleNamespace(central_bank=central_bank)})
    hook = create_policy_rate_shock_hook(country_code="FRA", initial_year=2020, time_unit=3, spec=spec)

    hook(simulation, 2020, 1)
    assert central_bank.functions["policy_rate"].compute_rate() == pytest.approx(0.02)
    hook(simulation, 2020, 4)
    assert central_bank.functions["policy_rate"].compute_rate() == pytest.approx(0.03)
    hook(simulation, 2020, 7)
    assert central_bank.functions["policy_rate"].compute_rate() == pytest.approx(0.02)


def test_tax_rate_shock_restores_baseline_after_duration():
    spec = ShockSpec(name="tax", kind="income_tax", period=0, magnitude=0.02, duration=1)
    government = SimpleNamespace(states={"Income Tax": 0.20})
    simulation = SimpleNamespace(countries={"FRA": SimpleNamespace(central_government=government)})
    hook = create_tax_rate_shock_hook(country_code="FRA", initial_year=2020, time_unit=3, spec=spec)

    hook(simulation, 2020, 1)
    assert government.states["Income Tax"] == pytest.approx(0.22)
    hook(simulation, 2020, 4)
    assert government.states["Income Tax"] == pytest.approx(0.20)


def test_government_consumption_shock_changes_exogenous_path_only_while_active():
    spec = ShockSpec(
        name="gov",
        kind="government_consumption",
        period=1,
        magnitude=0.10,
        duration=2,
        mode="multiplicative",
    )
    national_accounts = pd.DataFrame({"Real Government Consumption (Value)": [100.0, 110.0, 120.0]})
    country = SimpleNamespace(exogenous=SimpleNamespace(national_accounts_during=national_accounts))
    simulation = SimpleNamespace(countries={"FRA": country})
    hook = create_government_consumption_shock_hook(country_code="FRA", initial_year=2020, time_unit=3, spec=spec)

    hook(simulation, 2020, 1)
    np.testing.assert_allclose(national_accounts["Real Government Consumption (Value)"], [100.0, 110.0, 120.0])
    hook(simulation, 2020, 4)
    np.testing.assert_allclose(national_accounts["Real Government Consumption (Value)"], [100.0, 121.0, 132.0])
