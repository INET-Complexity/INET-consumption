"""Controlled separations must restrict hiring without changing unemployment status."""

from types import SimpleNamespace

import numpy as np
import pytest

from macromodel.agents.individuals.individual_properties import ActivityStatus as A
from macromodel.markets.labour_market.func.clearing import (
    DefaultLabourMarketClearer,
    PolednaLabourMarketClearer,
    ReservationWageBindingDefaultLabourMarketClearer,
)
from macromodel.utils.prehooks.irf_shocks import ShockSpec, create_unemployment_rate_shock_hook


def setup_market():
    values = dict(
        n_firms=1,
        normalised_labour_inputs=np.array([5.0]),
        labour_inputs=np.array([5.0]),
        desired_labour_inputs=np.array([10.0]),
    )
    firms = SimpleNamespace(
        ts=SimpleNamespace(current=values.__getitem__),
        states={
            "Employments": [list(range(5))],
            "Industry": np.array([0]),
            "Labour Productivity by Industry": np.ones(1),
            "offered_wage_function": lambda firm, labour: np.ones_like(labour) * 100,
        },
    )
    iv = dict(labour_inputs=np.ones(7), employee_income=np.ones(7) * 10, reservation_wages=np.zeros(7))
    individuals = SimpleNamespace(
        ts=SimpleNamespace(current=iv.__getitem__),
        states={
            "Activity Status": np.array([A.EMPLOYED] * 5 + [A.UNEMPLOYED] * 2),
            "Corresponding Firm ID": np.array([0] * 5 + [-1] * 2),
            "Employment Industry": np.zeros(7, dtype=int),
            "Corresponding Household ID": np.arange(7),
            "Started New Job": np.zeros(7, dtype=bool),
            "Offered Wage of Accepted Job": np.zeros(7),
        },
    )
    households = SimpleNamespace(ts=SimpleNamespace(current=lambda name: np.ones(7) * 100))
    sim = SimpleNamespace(random_seed=12, countries={"FRA": SimpleNamespace(individuals=individuals, firms=firms)})
    return sim, firms, individuals, households


def clearer(cls):
    return cls(
        hiring_speed=1.0,
        firing_speed=0.0,
        random_firing_probability=0.0,
        sorted_firing=False,
        optimised_hiring=False,
        allow_switching_industries=True,
        consider_reservation_wages=False,
        firing_cost_fraction=0.0,
        hiring_cost_fraction=0.0,
        individuals_quitting=False,
        individuals_quitting_temperature=0.0,
        compare_with_normalised_inputs=True,
        round_target_employment=False,
    )


@pytest.mark.parametrize(
    "cls", [DefaultLabourMarketClearer, ReservationWageBindingDefaultLabourMarketClearer, PolednaLabourMarketClearer]
)
def test_only_displaced_workers_blocked_then_eligible_at_window_end(cls):
    sim, firms, ind, hh = setup_market()
    spec = ShockSpec(name="jobs", kind="unemployment_rate", period=1, duration=4, magnitude=0.01, prevent_rehiring=True)
    hook = create_unemployment_rate_shock_hook(country_code="FRA", initial_year=2020, time_unit=3, spec=spec)
    hook(sim, 2020, 1)
    assert not ind.states["IRF Rehiring Blocked"].any()
    hook(sim, 2020, 4)
    blocked = ind.states["IRF Rehiring Blocked"].copy()
    assert blocked.sum() == 1
    for month in [4, 7, 10]:
        hook(sim, 2020, month)
        clearer(cls).clear(firms, hh, ind)
        mask = ind.states["IRF Rehiring Blocked"]
        assert np.all(ind.states["Activity Status"][mask] == A.UNEMPLOYED)
        assert np.all(ind.states["Corresponding Firm ID"][mask] == -1)
        assert np.all(ind.states["Activity Status"][5:] == A.EMPLOYED)
    hook(sim, 2021, 1)
    clearer(cls).clear(firms, hh, ind)
    blocked = ind.states["IRF Rehiring Blocked"].copy()
    assert blocked.sum() == 4  # one new separation each active quarter
    hook(sim, 2021, 1)  # duplicate calls do not separate twice
    np.testing.assert_array_equal(blocked, ind.states["IRF Rehiring Blocked"])
    hook(sim, 2021, 4)
    assert not ind.states["IRF Rehiring Blocked"].any()
    assert np.all(ind.states["Activity Status"][blocked] == A.UNEMPLOYED)
    clearer(cls).clear(firms, hh, ind)
    assert np.all(ind.states["Activity Status"][blocked] == A.EMPLOYED)


def test_disabled_option_preserves_draws_and_does_not_add_lockout_state():
    sim1, _, ind1, _ = setup_market()
    sim2, _, ind2, _ = setup_market()
    for sim, enabled in [(sim1, False), (sim2, True)]:
        hook = create_unemployment_rate_shock_hook(
            country_code="FRA",
            initial_year=2020,
            time_unit=3,
            spec=ShockSpec(name="jobs", kind="unemployment_rate", period=1, magnitude=0.01, prevent_rehiring=enabled),
        )
        hook(sim, 2020, 4)
    np.testing.assert_array_equal(ind1.states["Activity Status"], ind2.states["Activity Status"])
    assert "IRF Rehiring Blocked" not in ind1.states


@pytest.mark.parametrize(
    "kwargs", [dict(kind="policy_rate", prevent_rehiring=True), dict(kind="unemployment_rate", prevent_rehiring="true")]
)
def test_invalid_lockout_spec_rejected(kwargs):
    with pytest.raises(ValueError, match="prevent_rehiring"):
        ShockSpec(name="bad", period=1, magnitude=0.01, **kwargs)
