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


@pytest.mark.parametrize(
    ("magnitude", "duration", "expected_counts"),
    [
        (0.02, 1, [20, 20, 20, 20]),
        (0.005, 4, [5, 10, 15, 20]),
    ],
)
def test_alternative_design_cohort_sizes_and_common_release(magnitude, duration, expected_counts):
    sim, firms, ind, _ = setup_market()
    ind.states["Activity Status"] = np.array([A.EMPLOYED] * 800 + [A.UNEMPLOYED] * 200)
    ind.states["Corresponding Firm ID"] = np.array([0] * 800 + [-1] * 200)
    ind.states["Started New Job"] = np.zeros(1000, dtype=bool)
    ind.states["Offered Wage of Accepted Job"] = np.zeros(1000)
    firms.states["Employments"] = [list(range(800))]
    hook = create_unemployment_rate_shock_hook(
        country_code="FRA",
        initial_year=2020,
        time_unit=3,
        spec=ShockSpec(
            name="alternative",
            kind="unemployment_rate",
            period=1,
            magnitude=magnitude,
            duration=duration,
            prevent_rehiring=True,
            rehiring_lockout_duration=4,
        ),
    )
    for (year, month), expected in zip([(2020, 4), (2020, 7), (2020, 10), (2021, 1)], expected_counts):
        hook(sim, year, month)
        assert ind.states["IRF Rehiring Blocked"].sum() == expected
        assert (ind.states["Activity Status"] == A.UNEMPLOYED).sum() == 200 + expected
    hook(sim, 2021, 4)
    assert not ind.states["IRF Rehiring Blocked"].any()
    assert (ind.states["Activity Status"] == A.UNEMPLOYED).sum() == 220


@pytest.mark.parametrize("lockout", [0, 2, 3.5, True])
def test_invalid_lockout_duration_rejected(lockout):
    with pytest.raises(ValueError, match="rehiring_lockout_duration"):
        ShockSpec(
            name="invalid",
            kind="unemployment_rate",
            period=1,
            magnitude=0.005,
            duration=4,
            prevent_rehiring=True,
            rehiring_lockout_duration=lockout,
        )


def test_lockout_duration_requires_restriction():
    with pytest.raises(ValueError, match="requires prevent_rehiring"):
        ShockSpec(name="invalid", kind="unemployment_rate", period=1, magnitude=0.02, rehiring_lockout_duration=4)


@pytest.mark.parametrize(
    "cls", [DefaultLabourMarketClearer, ReservationWageBindingDefaultLabourMarketClearer, PolednaLabourMarketClearer]
)
def test_one_off_workers_stay_blocked_after_separation_window_ends(cls):
    sim, firms, ind, hh = setup_market()
    hook = create_unemployment_rate_shock_hook(
        country_code="FRA",
        initial_year=2020,
        time_unit=3,
        spec=ShockSpec(
            name="once",
            kind="unemployment_rate",
            period=1,
            magnitude=0.01,
            duration=1,
            prevent_rehiring=True,
            rehiring_lockout_duration=4,
        ),
    )
    for year, month in [(2020, 4), (2020, 7), (2020, 10), (2021, 1)]:
        hook(sim, year, month)
        clearer(cls).clear(firms, hh, ind)
        mask = ind.states["IRF Rehiring Blocked"]
        assert mask.sum() == 1
        assert np.all(ind.states["Activity Status"][mask] == A.UNEMPLOYED)
    blocked = mask.copy()
    hook(sim, 2021, 4)
    clearer(cls).clear(firms, hh, ind)
    assert np.all(ind.states["Activity Status"][blocked] == A.EMPLOYED)


@pytest.mark.parametrize(
    "cls", [DefaultLabourMarketClearer, ReservationWageBindingDefaultLabourMarketClearer, PolednaLabourMarketClearer]
)
@pytest.mark.parametrize("duration", [1, 4])
def test_all_unemployed_scope_blocks_replacement_and_restores_hiring(cls, duration):
    sim, firms, ind, hh = setup_market()
    # Reflect workforce changes between periods as the simulation would.
    current = firms.ts.current
    firms.ts.current = lambda key: (
        np.array([len(firms.states["Employments"][0])], dtype=float)
        if key in {"labour_inputs", "normalised_labour_inputs"}
        else current(key)
    )
    hook = create_unemployment_rate_shock_hook(
        country_code="FRA",
        initial_year=2020,
        time_unit=3,
        spec=ShockSpec(
            name="macro",
            kind="unemployment_rate",
            period=1,
            magnitude=0.01,
            duration=duration,
            prevent_rehiring=True,
            rehiring_lockout_duration=4,
            rehiring_scope="all_unemployed",
        ),
    )
    hook(sim, 2020, 1)
    assert not ind.states["IRF Rehiring Blocked"].any()
    for year, month in [(2020, 4), (2020, 7), (2020, 10), (2021, 1)]:
        hook(sim, year, month)
        assert ind.states["IRF Rehiring Blocked"].all()
        if duration == 1 and month == 7:
            # Separation after the prehook: this worker must also stay out of hiring.
            worker = firms.states["Employments"][0].pop()
            ind.states["Activity Status"][worker] = A.UNEMPLOYED
            ind.states["Corresponding Firm ID"][worker] = -1
        unemployed = ind.states["Activity Status"] == A.UNEMPLOYED
        result = clearer(cls).clear(firms, hh, ind)
        assert result[1] == 0
        assert np.all(ind.states["Activity Status"][unemployed] == A.UNEMPLOYED)
        assert np.all(ind.states["Corresponding Firm ID"][unemployed] == -1)
    # Expiry restores the pool, without setting anyone employed in the hook.
    unemployed = ind.states["Activity Status"] == A.UNEMPLOYED
    hook(sim, 2021, 4)
    assert not ind.states["IRF Rehiring Blocked"].any()
    assert np.all(ind.states["Activity Status"][unemployed] == A.UNEMPLOYED)
    clearer(cls).clear(firms, hh, ind)
    assert np.all(ind.states["Activity Status"][unemployed] == A.EMPLOYED)


def test_all_unemployed_mask_preserves_job_to_job_matching():
    matcher = clearer(ReservationWageBindingDefaultLabourMarketClearer)
    assignments = np.array([0, 0, 1, 1, -1])
    statuses = np.array([A.EMPLOYED] * 4 + [A.UNEMPLOYED])
    employees = [[0, 1], [2, 3]]
    _, joined = matcher.hiring(
        firm_employments=employees,
        firm_industries=np.zeros(2, dtype=int),
        current_individuals_activity=statuses,
        current_individuals_industry=np.zeros(5, dtype=int),
        individuals_corresponding_firm=assignments,
        prev_individuals_productivity=np.ones(5),
        desired_labour_inputs=np.array([3.0, 2.0]),
        prev_labour_inputs=np.array([2.0, 2.0]),
        offered_wage_function=lambda firm, labour: np.ones_like(labour) * 100,
        offered_wage=np.zeros(5),
        individual_reservation_wages=np.zeros(5),
        current_individual_wages=np.ones(5) * 10,
        average_industry_productivity=np.ones(1),
        current_individual_started_new_job=np.zeros(5, dtype=bool),
        searchers=np.array([False, False, True, False, False]),
        searcher_stay_rate=np.array([np.inf, np.inf, 1.0, np.inf, np.inf]),
        offered_rate_by_firm=np.array([100.0, 0.0]),
        rehiring_blocked=np.ones(5, dtype=bool),
    )
    assert joined == 0
    assert matcher.last_num_job_to_job == 1
    assert assignments[2] == 0
    assert assignments[4] == -1
    assert statuses[4] == A.UNEMPLOYED


@pytest.mark.parametrize("scope", ["invalid", "ALL_UNEMPLOYED"])
def test_invalid_rehiring_scope_rejected(scope):
    with pytest.raises(ValueError, match="rehiring_scope"):
        ShockSpec(
            name="invalid",
            kind="unemployment_rate",
            period=1,
            magnitude=0.02,
            prevent_rehiring=True,
            rehiring_scope=scope,
        )


def test_all_unemployed_scope_requires_restriction():
    with pytest.raises(ValueError, match="requires prevent_rehiring"):
        ShockSpec(name="invalid", kind="unemployment_rate", period=1, magnitude=0.02, rehiring_scope="all_unemployed")
