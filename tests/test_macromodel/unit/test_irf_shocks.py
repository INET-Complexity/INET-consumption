from types import SimpleNamespace

import numpy as np
import pytest

from macromodel.agents.central_bank.func.policy_rate import SmoothTaylorRule
from macromodel.agents.government_entities.func.consumption import ExpectedGrowthGovernmentConsumptionSetter
from macromodel.agents.individuals.individual_properties import ActivityStatus
from macromodel.configurations import CountryConfiguration, SimulationConfiguration
from macromodel.simulation import Simulation
from macromodel.utils.prehooks.irf_shocks import (
    ShockSpec,
    create_government_consumption_shock_hook,
    create_irf_shock_hook,
    create_policy_rate_shock_hook,
    create_tax_rate_shock_hook,
    create_unemployment_rate_shock_hook,
)


class DummyPolicyRate:
    def compute_rate(self, *, shock=0.0, **_kwargs):
        return 0.02


class DummyGovernmentConsumptionSetter:
    def compute_target_consumption(self, **_kwargs):
        return np.array([20.0, 30.0, 50.0])


def _compute_government_target(setter):
    return setter.compute_target_consumption(
        previous_desired_government_consumption=np.array([20.0, 30.0, 50.0]),
        model=None,
        historic_total_consumption=None,
        initial_good_prices=np.ones(3),
        current_good_prices=np.array([2.0, 1.0, 3.0]),
        expected_growth=0.02,
        expected_inflation=0.05,
        current_time=1,
        exogenous_total_consumption=None,
        forecasting_window=1,
    )


def _government_consumption_simulation():
    setter = ExpectedGrowthGovernmentConsumptionSetter(
        consistency=1.0,
        default_growth=0.99,
        sectoral_weights="initial_fixed",
    )
    government_entities = SimpleNamespace(functions={"consumption": setter})
    country = SimpleNamespace(government_entities=government_entities)
    return SimpleNamespace(countries={"FRA": country}), setter


@pytest.mark.parametrize("name", ["../bad", "/bad", ".hidden", "bad/name", "bad name"])
def test_shock_spec_rejects_unsafe_names(name):
    with pytest.raises(ValueError, match="ShockSpec.name"):
        ShockSpec(name=name, kind="policy_rate", period=0, magnitude=0.01)


@pytest.mark.parametrize("magnitude", [np.nan, np.inf, -np.inf, [0.01, 0.02]])
def test_shock_spec_rejects_nonfinite_or_nonscalar_magnitude(magnitude):
    with pytest.raises(ValueError, match="ShockSpec.magnitude must be a finite scalar"):
        ShockSpec(name="invalid-magnitude", kind="policy_rate", period=0, magnitude=magnitude)


@pytest.mark.parametrize("kind", ["common_income_uncertainty", "individual_income_uncertainty"])
def test_income_uncertainty_shock_is_reproducible_and_clears_after_window(kind):
    spec = ShockSpec(
        name="income-variance",
        kind=kind,
        period=1,
        magnitude=0.10,
        duration=1,
        mode="multiplicative",
        random_log_sd=0.25,
    )

    def simulation(seed=12):
        households = SimpleNamespace(
            ts=SimpleNamespace(current=lambda _name: 4096),
            states={},
        )
        country = SimpleNamespace(households=households)
        return SimpleNamespace(countries={"FRA": country}, random_seed=seed)

    first = simulation()
    second = simulation()
    hook = create_irf_shock_hook(country_code="FRA", initial_year=2020, time_unit=3, spec=spec)

    hook(first, 2020, 4)
    hook(second, 2020, 4)
    first_factors = first.countries["FRA"].households.states["income_uncertainty_shock_factors"]
    second_factors = second.countries["FRA"].households.states["income_uncertainty_shock_factors"]
    assert first_factors["period"] == 2
    assert first_factors["common"] == second_factors["common"]
    np.testing.assert_array_equal(first_factors["individual"], second_factors["individual"])
    assert np.all(np.asarray(first_factors["individual"]) > 0.0)

    if kind == "common_income_uncertainty":
        assert first_factors["common"] != 1.0
        np.testing.assert_array_equal(first_factors["individual"], np.ones(4096))
        common_draws = []
        for seed in range(1024):
            sample = simulation(seed=seed)
            hook(sample, 2020, 4)
            common_draws.append(
                sample.countries["FRA"].households.states["income_uncertainty_shock_factors"]["common"]
            )
        log_draws = np.log(common_draws)
        assert np.mean(log_draws) == pytest.approx(np.log1p(0.10) - 0.5 * 0.25**2, abs=0.02)
        assert np.std(log_draws) == pytest.approx(0.25, abs=0.02)
    else:
        assert first_factors["common"] == 1.0
        assert np.unique(first_factors["individual"]).size > 1
        log_draws = np.log(first_factors["individual"])
        assert np.mean(log_draws) == pytest.approx(np.log1p(0.10) - 0.5 * 0.25**2, abs=0.01)
        assert np.std(log_draws) == pytest.approx(0.25, abs=0.01)

    hook(first, 2020, 7)
    cleared = first.countries["FRA"].households.states["income_uncertainty_shock_factors"]
    assert cleared["period"] == 3
    assert cleared["common"] == 1.0
    np.testing.assert_array_equal(cleared["individual"], np.ones(4096))


@pytest.mark.parametrize(
    ("kind", "overrides"),
    [
        ("common_income_uncertainty", {"mode": "additive"}),
        ("individual_income_uncertainty", {"random_log_sd": None}),
        ("individual_income_uncertainty", {"period": 0}),
    ],
)
def test_income_uncertainty_shock_rejects_invalid_specs(kind, overrides):
    values = dict(
        name="invalid-uncertainty",
        kind=kind,
        period=1,
        magnitude=0.10,
        mode="multiplicative",
        random_log_sd=0.25,
    )
    values.update(overrides)
    spec = ShockSpec(**values)

    with pytest.raises(ValueError, match="Income-uncertainty"):
        create_irf_shock_hook(country_code="FRA", initial_year=2020, time_unit=3, spec=spec)


@pytest.mark.parametrize("kind", ["common_income_uncertainty", "individual_income_uncertainty"])
def test_income_uncertainty_shock_changes_full_simulation_household_demand(datawrapper, kind):
    spec = ShockSpec(
        name="uncertainty-irf",
        kind=kind,
        period=1,
        magnitude=0.50,
        mode="multiplicative",
        random_log_sd=0.10,
    )

    def run_arm(shock_spec=None):
        country_configuration = CountryConfiguration()
        consumption_config = country_configuration.households.functions.consumption
        consumption_config.name = "CreditAugmentedConsumption"
        consumption_config.parameters.update(
            uses_income_belief_learning=True,
            income_belief_learning_horizon={"delta": 0.95, "S": 40},
            permanent_income_propensity=0.1,
            uncertainty_propensity=0.1,
        )
        configuration = SimulationConfiguration(
            country_configurations={"FRA": country_configuration},
            seed=12,
            t_max=2,
        )
        model = Simulation.from_datawrapper(datawrapper=datawrapper, simulation_configuration=configuration)
        country = model.countries["FRA"]
        country._common_permanent_income_terms = lambda: (0.0, 0.04)
        households = country.households
        learning_inputs_by_period = {}
        compute_inputs = households.current_income_belief_learning_inputs

        def capture_learning_inputs(**kwargs):
            inputs = compute_inputs(**kwargs)
            period = len(households.ts.historic("income"))
            learning_inputs_by_period.setdefault(
                period,
                {name: np.asarray(value).copy() for name, value in inputs.items()},
            )
            return inputs

        households.current_income_belief_learning_inputs = capture_learning_inputs
        if shock_spec is not None:
            model.prehooks.append(
                create_irf_shock_hook(
                    country_code="FRA",
                    initial_year=datawrapper.configuration.year,
                    time_unit=datawrapper.time_unit,
                    spec=shock_spec,
                )
            )
        model.run()
        return learning_inputs_by_period, households.functions["consumption"].last_target_consumption_components

    baseline_inputs, baseline_demand = run_arm()
    shocked_inputs, shocked_demand = run_arm(spec)

    assert 2 in baseline_inputs and 2 in shocked_inputs
    assert np.any(np.abs(shocked_inputs[2]["uncertainty_delta"] - baseline_inputs[2]["uncertainty_delta"]) > 1e-12)
    np.testing.assert_array_equal(
        shocked_demand["target_consumption_uncertainty_delta"],
        shocked_inputs[2]["uncertainty_delta"],
    )
    assert np.any(
        np.abs(
            shocked_demand["target_consumption_uncertainty"]
            - baseline_demand["target_consumption_uncertainty"]
        )
        > 1e-12
    )


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


@pytest.mark.parametrize(
    ("lower_bound", "magnitude", "time_unit", "expected_rate"),
    [
        (0.0, -0.01, 3, 0.0),
        (0.0, -0.01, 1, 0.0),
        (0.01, -0.02, 3, 0.0025),
        (-0.01, -0.02, 3, -0.0025),
        (None, -0.01, 3, -0.0025),
        (0.0, 0.01, 3, 0.0025),
    ],
)
@pytest.mark.parametrize("positional", [False, True])
def test_policy_rate_shock_respects_annual_floor(lower_bound, magnitude, time_unit, expected_rate, positional):
    rule = SmoothTaylorRule(lower_bound=lower_bound)
    central_bank = SimpleNamespace(functions={"policy_rate": rule})
    simulation = SimpleNamespace(countries={"FRA": SimpleNamespace(central_bank=central_bank)})
    hook = create_policy_rate_shock_hook(
        country_code="FRA",
        initial_year=2020,
        time_unit=time_unit,
        spec=ShockSpec(name="bounded_rate", kind="policy_rate", period=0, magnitude=magnitude),
    )
    states = dict(rho=0.0, r_star=-0.02, targeted_inflation_rate=0.02, phi_pi=0.0, phi_q=0.0)
    inputs = dict(
        prev_rate=0.0,
        inflation=0.0,
        growth=0.0,
        central_bank_states=states,
        cpi_yoy_inflation=0.0,
        output_gap=0.0,
        time_unit=time_unit,
    )
    hook(simulation, 2020, 1)
    shocked = central_bank.functions["policy_rate"]
    actual = shocked.compute_rate(*inputs.values()) if positional else shocked.compute_rate(**inputs)
    assert actual == pytest.approx(expected_rate)

    hook(simulation, 2020, 1 + time_unit)
    assert shocked.compute_rate(**inputs) == pytest.approx(rule.compute_rate(**inputs))


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


@pytest.mark.parametrize(
    ("baseline", "magnitude"),
    [(0.995, 0.01), (0.005, -0.01), ([0.20, 0.995], 0.01)],
)
def test_tax_rate_shock_rejects_rates_outside_unit_interval(baseline, magnitude):
    spec = ShockSpec(name="invalid-tax", kind="income_tax", period=0, magnitude=magnitude)
    original = np.asarray(baseline, dtype=float)
    stored_baseline = float(original) if original.ndim == 0 else original.copy()
    government = SimpleNamespace(states={"Income Tax": stored_baseline})
    simulation = SimpleNamespace(countries={"FRA": SimpleNamespace(central_government=government)})
    hook = create_tax_rate_shock_hook(country_code="FRA", initial_year=2020, time_unit=3, spec=spec)

    with pytest.raises(ValueError, match=r"tax rates outside \[0, 1\]"):
        hook(simulation, 2020, 1)

    if original.ndim == 0:
        assert government.states["Income Tax"] == pytest.approx(float(original))
    else:
        np.testing.assert_array_equal(government.states["Income Tax"], original)


def test_government_consumption_shock_adds_to_default_target_setter():
    spec = ShockSpec(name="gov", kind="government_consumption", period=0, magnitude=10.0)
    simulation, setter = _government_consumption_simulation()
    hook = create_government_consumption_shock_hook(country_code="FRA", initial_year=2020, time_unit=3, spec=spec)

    baseline = _compute_government_target(setter)
    hook(simulation, 2020, 1)
    shocked = _compute_government_target(simulation.countries["FRA"].government_entities.functions["consumption"])

    np.testing.assert_allclose(shocked.sum() - baseline.sum(), 10.0)
    np.testing.assert_allclose(shocked / shocked.sum(), baseline / baseline.sum())


def test_government_consumption_additive_shock_supports_public_setter_contract_only():
    spec = ShockSpec(name="gov", kind="government_consumption", period=0, magnitude=10.0)
    government_entities = SimpleNamespace(functions={"consumption": DummyGovernmentConsumptionSetter()})
    simulation = SimpleNamespace(countries={"FRA": SimpleNamespace(government_entities=government_entities)})
    hook = create_government_consumption_shock_hook(country_code="FRA", initial_year=2020, time_unit=3, spec=spec)

    hook(simulation, 2020, 1)
    shocked = _compute_government_target(simulation.countries["FRA"].government_entities.functions["consumption"])

    np.testing.assert_allclose(shocked, [22.0, 33.0, 55.0])


def test_government_consumption_shock_scales_default_target_setter():
    spec = ShockSpec(
        name="gov",
        kind="government_consumption",
        period=0,
        magnitude=0.10,
        mode="multiplicative",
    )
    simulation, setter = _government_consumption_simulation()
    hook = create_government_consumption_shock_hook(country_code="FRA", initial_year=2020, time_unit=3, spec=spec)

    baseline = _compute_government_target(setter)
    hook(simulation, 2020, 1)
    shocked = _compute_government_target(simulation.countries["FRA"].government_entities.functions["consumption"])

    np.testing.assert_allclose(shocked, baseline * 1.10)


def test_government_consumption_shock_window_only_affects_active_period():
    spec = ShockSpec(name="gov", kind="government_consumption", period=1, magnitude=10.0)
    simulation, setter = _government_consumption_simulation()
    hook = create_government_consumption_shock_hook(country_code="FRA", initial_year=2020, time_unit=3, spec=spec)

    baseline = _compute_government_target(setter)
    hook(simulation, 2020, 1)
    inactive = _compute_government_target(simulation.countries["FRA"].government_entities.functions["consumption"])
    hook(simulation, 2020, 4)
    active = _compute_government_target(simulation.countries["FRA"].government_entities.functions["consumption"])

    np.testing.assert_allclose(inactive, baseline)
    assert active.sum() > baseline.sum()


def test_government_consumption_negative_additive_shock_clamps_at_zero():
    spec = ShockSpec(name="gov", kind="government_consumption", period=0, magnitude=-200.0)
    simulation, _setter = _government_consumption_simulation()
    hook = create_government_consumption_shock_hook(country_code="FRA", initial_year=2020, time_unit=3, spec=spec)

    hook(simulation, 2020, 1)
    shocked = _compute_government_target(simulation.countries["FRA"].government_entities.functions["consumption"])

    np.testing.assert_allclose(shocked, np.zeros(3))


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
