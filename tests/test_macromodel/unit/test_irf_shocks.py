from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from macromodel.agents.individuals.individual_properties import ActivityStatus
from macromodel.configurations import CountryConfiguration, SimulationConfiguration
from macromodel.simulation import Simulation
from macromodel.utils.prehooks.irf_shocks import (
    ShockSpec,
    create_government_consumption_shock_hook,
    create_policy_rate_shock_hook,
    create_tax_rate_shock_hook,
    create_unemployment_rate_shock_hook,
)


class DummyPolicyRate:
    def compute_rate(self, *, shock=0.0, **_kwargs):
        return 0.02


def test_policy_rate_shock_converts_annual_rate_points_to_period_rate_points():
    spec = ShockSpec(name="rate", kind="policy_rate", period=1, magnitude=0.01)
    central_bank = SimpleNamespace(functions={"policy_rate": DummyPolicyRate()})
    simulation = SimpleNamespace(countries={"FRA": SimpleNamespace(central_bank=central_bank)})
    hook = create_policy_rate_shock_hook(country_code="FRA", initial_year=2020, time_unit=3, spec=spec)

    hook(simulation, 2020, 1)
    assert central_bank.functions["policy_rate"].compute_rate() == pytest.approx(0.02)
    hook(simulation, 2020, 4)
    assert central_bank.functions["policy_rate"].compute_rate() == pytest.approx(0.0225)
    hook(simulation, 2020, 7)
    assert central_bank.functions["policy_rate"].compute_rate() == pytest.approx(0.02)


def test_policy_rate_shock_perturbs_real_simulation_policy_rate(datawrapper):
    spec = ShockSpec(name="rate", kind="policy_rate", period=0, magnitude=0.01)
    configuration = SimulationConfiguration(
        country_configurations={"FRA": CountryConfiguration()},
        seed=12,
        t_max=2,
    )
    baseline = Simulation.from_datawrapper(datawrapper=datawrapper, simulation_configuration=configuration)
    shocked = Simulation.from_datawrapper(datawrapper=datawrapper, simulation_configuration=configuration)
    shocked.prehooks.append(
        create_policy_rate_shock_hook(
            country_code="FRA",
            initial_year=datawrapper.configuration.year,
            time_unit=datawrapper.time_unit,
            spec=spec,
        )
    )

    baseline.run()
    shocked.run()

    baseline_policy_rate = baseline.countries["FRA"].central_bank.ts.historic("policy_rate")[1][0]
    shocked_policy_rate = shocked.countries["FRA"].central_bank.ts.historic("policy_rate")[1][0]
    assert shocked_policy_rate - baseline_policy_rate == pytest.approx(0.01 / 4.0)


def test_tax_rate_shock_restores_baseline_after_duration():
    spec = ShockSpec(name="tax", kind="income_tax", period=0, magnitude=0.02, duration=1)
    government = SimpleNamespace(states={"Income Tax": 0.20})
    simulation = SimpleNamespace(countries={"FRA": SimpleNamespace(central_government=government)})
    hook = create_tax_rate_shock_hook(country_code="FRA", initial_year=2020, time_unit=3, spec=spec)

    hook(simulation, 2020, 1)
    assert government.states["Income Tax"] == pytest.approx(0.22)
    hook(simulation, 2020, 4)
    assert government.states["Income Tax"] == pytest.approx(0.20)


def test_government_consumption_shock_targets_realised_response_rows():
    spec = ShockSpec(
        name="gov",
        kind="government_consumption",
        period=1,
        magnitude=0.10,
        duration=2,
        mode="multiplicative",
    )
    national_accounts = pd.DataFrame({"Real Government Consumption (Value)": [100.0, 110.0, 120.0, 130.0]})
    country = SimpleNamespace(exogenous=SimpleNamespace(national_accounts_during=national_accounts))
    simulation = SimpleNamespace(countries={"FRA": country})
    hook = create_government_consumption_shock_hook(country_code="FRA", initial_year=2020, time_unit=3, spec=spec)

    hook(simulation, 2020, 1)
    np.testing.assert_allclose(national_accounts["Real Government Consumption (Value)"], [100.0, 110.0, 120.0, 130.0])
    hook(simulation, 2020, 4)
    np.testing.assert_allclose(national_accounts["Real Government Consumption (Value)"], [100.0, 110.0, 132.0, 143.0])


def test_unemployment_rate_shock_separates_sampled_employed_workers():
    spec = ShockSpec(name="unemp", kind="unemployment_rate", period=0, magnitude=0.25)
    individuals = SimpleNamespace(
        states={
            "Activity Status": np.array(
                [
                    ActivityStatus.EMPLOYED,
                    ActivityStatus.EMPLOYED,
                    ActivityStatus.EMPLOYED,
                    ActivityStatus.UNEMPLOYED,
                ],
                dtype=object,
            ),
            "Corresponding Firm ID": np.array([0, 0, 1, -1]),
            "Started New Job": np.array([False, False, False, False]),
            "Offered Wage of Accepted Job": np.array([10.0, 10.0, 10.0, 0.0]),
        }
    )
    firms = SimpleNamespace(states={"Employments": [[0, 1], [2]]}, ts=SimpleNamespace(current=lambda _name: [2]))
    country = SimpleNamespace(individuals=individuals, firms=firms)
    simulation = SimpleNamespace(countries={"FRA": country}, random_seed=12)
    hook = create_unemployment_rate_shock_hook(country_code="FRA", initial_year=2020, time_unit=3, spec=spec)

    hook(simulation, 2020, 1)

    unemployed = individuals.states["Activity Status"] == ActivityStatus.UNEMPLOYED
    assert unemployed.sum() == 2
    assert individuals.states["Corresponding Firm ID"][unemployed].tolist() == [-1, -1]
    assert len(firms.states["Employments"][0]) == 1
    assert firms.states["Employments"][1] == [2]


def test_unemployment_rate_shock_accepts_scalar_n_firms():
    spec = ShockSpec(name="unemp", kind="unemployment_rate", period=0, magnitude=0.25)
    individuals = SimpleNamespace(
        states={
            "Activity Status": np.array(
                [
                    ActivityStatus.EMPLOYED,
                    ActivityStatus.EMPLOYED,
                    ActivityStatus.EMPLOYED,
                    ActivityStatus.UNEMPLOYED,
                ],
                dtype=object,
            ),
            "Corresponding Firm ID": np.array([0, 0, 1, -1]),
            "Started New Job": np.array([False, False, False, False]),
            "Offered Wage of Accepted Job": np.array([10.0, 10.0, 10.0, 0.0]),
        }
    )
    firms = SimpleNamespace(states={"Employments": [[0, 1], [2]]}, ts=SimpleNamespace(current=lambda _name: 2))
    country = SimpleNamespace(individuals=individuals, firms=firms)
    simulation = SimpleNamespace(countries={"FRA": country}, random_seed=12)
    hook = create_unemployment_rate_shock_hook(country_code="FRA", initial_year=2020, time_unit=3, spec=spec)

    hook(simulation, 2020, 1)

    assert np.sum(individuals.states["Activity Status"] == ActivityStatus.UNEMPLOYED) == 2
