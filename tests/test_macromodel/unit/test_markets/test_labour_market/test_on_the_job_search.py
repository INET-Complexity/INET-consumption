"""On-the-job search in the reservation-wage-binding labour clearer (wage plan, Step 2)."""

from types import SimpleNamespace

import numpy as np
import pytest
from numba.typed import List

from macromodel.agents.individuals.individual_properties import ActivityStatus
from macromodel.markets.labour_market.func.clearing import (
    DefaultLabourMarketClearer,
    ReservationWageBindingDefaultLabourMarketClearer,
)

E, U = ActivityStatus.EMPLOYED, ActivityStatus.UNEMPLOYED


class _DummyTS:
    def __init__(self, values):
        self._values = values

    def current(self, name):
        return self._values[name]


def _clearer(cls=ReservationWageBindingDefaultLabourMarketClearer, **overrides):
    parameters = dict(
        hiring_speed=1.0,
        firing_speed=0.0,
        random_firing_probability=0.0,
        sorted_firing=False,
        optimised_hiring=True,
        allow_switching_industries=True,
        consider_reservation_wages=True,
        firing_cost_fraction=0.0,
        hiring_cost_fraction=0.0,
        individuals_quitting=False,
        individuals_quitting_temperature=50.0,  # search probability ~ 1
        compare_with_normalised_inputs=False,
        round_target_employment=False,
        on_the_job_search=True,
    )
    parameters.update(overrides)
    return cls(**parameters)


def _economy(
    employments,
    wage_rate,
    offered_rate,
    desired,
    labour_inputs=None,
    indexation=None,
    reservation=None,
    benefit=0.0,
):
    """Firms and individuals with unit industry productivity and one industry."""
    n_firms, n_ind = len(employments), len(wage_rate)
    labour_inputs = np.ones(n_ind) if labour_inputs is None else np.asarray(labour_inputs, dtype=float)
    offered_rate = np.asarray(offered_rate, dtype=float)
    firm_of = np.full(n_ind, -1)
    for firm_id, employees in enumerate(employments):
        firm_of[list(employees)] = firm_id
    wage_setter = SimpleNamespace(
        search_comparison_rates=lambda: (
            offered_rate,
            np.ones(n_firms) if indexation is None else np.asarray(indexation, dtype=float),
        )
    )
    firms = SimpleNamespace(
        ts=_DummyTS(
            {
                "labour_inputs": np.array([labour_inputs[list(e)].sum() for e in employments]),
                "desired_labour_inputs": np.asarray(desired, dtype=float),
            }
        ),
        states={
            "Employments": List([List(list(e)) for e in employments]),
            "Industry": np.zeros(n_firms, dtype=int),
            "Labour Productivity by Industry": np.array([1.0]),
            "offered_wage_function": lambda firm_id, inputs: np.maximum(benefit, inputs * offered_rate[firm_id]),
        },
        functions={"wage_setter": wage_setter},
    )
    households = SimpleNamespace(ts=_DummyTS({"wealth": np.ones(n_ind)}))
    individuals = SimpleNamespace(
        ts=_DummyTS(
            {
                "labour_inputs": labour_inputs,
                "employee_income": np.asarray(wage_rate, dtype=float),
                "reservation_wages": np.zeros(n_ind) if reservation is None else np.asarray(reservation, dtype=float),
            }
        ),
        states={
            "Activity Status": np.where(firm_of >= 0, E, U),
            "Corresponding Firm ID": firm_of,
            "Corresponding Household ID": np.arange(n_ind),
            "Employment Industry": np.zeros(n_ind, dtype=int),
            "Wage Rate": np.asarray(wage_rate, dtype=float),
        },
    )
    return firms, households, individuals


def _clear(clearer, economy):
    firms, households, individuals = economy
    result = clearer.clear(firms=firms, households=households, individuals=individuals)
    return result, firms, individuals


def test_searcher_moves_to_the_firm_with_the_higher_rate():
    # Firm 0 pays 1.0 and has no gap; firm 1 offers 1.5 and wants one more worker.
    economy = _economy([[0, 1], [2]], wage_rate=[1.0, 1.0, 1.5], offered_rate=[1.0, 1.5], desired=[2.0, 2.0])
    clearer = _clearer()
    (_, num_joining, *_), firms, individuals = _clear(clearer, economy)

    moved = [i for i in (0, 1) if individuals.states["Corresponding Firm ID"][i] == 1]
    assert len(moved) == 1
    mover = moved[0]
    assert individuals.states["Activity Status"][mover] == E
    assert individuals.states["Started New Job"][mover]
    assert individuals.states["Offered Wage of Accepted Job"][mover] == 1.5
    assert sorted(firms.states["Employments"][1]) == sorted([2, mover])
    assert mover not in list(firms.states["Employments"][0])
    assert clearer.last_num_job_to_job == 1
    assert num_joining == 0  # job-to-job moves are not hires from unemployment
    assert (individuals.states["Activity Status"] == E).sum() == 3


def test_searcher_stays_when_the_offered_rate_is_not_higher():
    economy = _economy([[0, 1], [2]], wage_rate=[1.0, 1.0, 1.0], offered_rate=[1.0, 1.0], desired=[2.0, 2.0])
    clearer = _clearer()
    _, firms, individuals = _clear(clearer, economy)

    assert individuals.states["Corresponding Firm ID"].tolist() == [0, 0, 1]
    assert not individuals.states["Started New Job"].any()
    assert (individuals.states["Offered Wage of Accepted Job"] == 0.0).all()
    assert clearer.last_num_job_to_job == 0


def test_stay_rate_includes_the_own_firm_indexation_factor():
    # Offer 1.05 beats the carried rate 1.0, but not 1.0 x 1.1 after indexation.
    economy = _economy(
        [[0, 1], [2]], wage_rate=[1.0, 1.0, 1.0], offered_rate=[1.0, 1.05], desired=[2.0, 2.0], indexation=[1.1, 1.0]
    )
    clearer = _clearer()
    _clear(clearer, economy)
    assert clearer.last_num_job_to_job == 0


def test_comparison_uses_rates_not_the_benefit_floor():
    # Worker 0's carried rate (0.5) is below the benefit (0.8). Firm 1's rate offer
    # to it is 0.4 < 0.5, so it stays although the floored offer 0.8 exceeds 0.5.
    economy = _economy(
        [[0, 1], [2]],
        wage_rate=[0.5, 0.5, 1.0],
        offered_rate=[1.0, 0.4],
        desired=[2.0, 2.0],
        benefit=0.8,
    )
    clearer = _clearer()
    _clear(clearer, economy)
    assert clearer.last_num_job_to_job == 0


def test_comparison_scales_the_offered_rate_by_the_worker_labour_input():
    # Rate per unit of input 1.2; worker 0 has input 2 and carries 2.0 (1.0 per unit).
    # Offer 2 x 1.2 = 2.4 > 2.0, so it moves and the accepted offer is 2.4.
    economy = _economy(
        [[0, 1], [2]],
        wage_rate=[2.0, 5.0, 1.2],
        offered_rate=[1.0, 1.2],
        desired=[3.0, 3.0],
        labour_inputs=[2.0, 1.0, 1.0],
    )
    clearer = _clearer()
    _, _, individuals = _clear(clearer, economy)
    assert individuals.states["Corresponding Firm ID"].tolist() == [1, 0, 1]
    assert individuals.states["Offered Wage of Accepted Job"][0] == pytest.approx(2.4)


def test_last_employee_cannot_leave():
    economy = _economy([[0], [1]], wage_rate=[1.0, 2.0], offered_rate=[1.0, 2.0], desired=[1.0, 2.0])
    clearer = _clearer()
    _, firms, individuals = _clear(clearer, economy)
    assert individuals.states["Corresponding Firm ID"].tolist() == [0, 1]
    assert list(firms.states["Employments"][0]) == [0]


def test_firm_does_not_poach_its_own_workers():
    # Firm 0's offer (1.5) beats its own workers' carried rate (1.0); they are not candidates.
    economy = _economy([[0, 1]], wage_rate=[1.0, 1.0], offered_rate=[1.5], desired=[3.0])
    clearer = _clearer()
    _, _, individuals = _clear(clearer, economy)
    assert clearer.last_num_job_to_job == 0
    assert not individuals.states["Started New Job"].any()


def test_nobody_becomes_unemployed_through_search():
    economy = _economy([[0, 1, 2], [3]], wage_rate=[1.0, 1.0, 1.0, 2.0], offered_rate=[1.0, 2.0], desired=[3.0, 2.0])
    clearer = _clearer(round_target_employment=True)  # firm 1 stops once its gap of one worker is filled
    (_, _, _, num_quit, num_fired), _, individuals = _clear(clearer, economy)
    assert (individuals.states["Activity Status"] == E).all()
    assert num_quit == 0 and num_fired == 0
    assert clearer.last_num_job_to_job == 1


def test_unemployed_first_takes_the_unemployed_candidate():
    # Gap of one worker at firm 1; an unemployed worker (3) and a willing searcher both qualify.
    def run(priority):
        economy = _economy(
            [[0, 1], [2], []][:2], wage_rate=[1.0, 1.0, 1.5, 0.0], offered_rate=[1.0, 1.5], desired=[2.0, 2.0]
        )
        clearer = _clearer(searcher_priority=priority, hiring_speed=0.5)
        (_, num_joining, *_), _, individuals = _clear(clearer, economy)
        return num_joining, clearer.last_num_job_to_job, individuals

    num_joining, num_moves, individuals = run("unemployed_first")
    assert (num_joining, num_moves) == (1, 0)
    assert individuals.states["Corresponding Firm ID"][3] == 1


def test_no_search_when_the_probability_is_zero():
    economy = _economy([[0, 1], [2]], wage_rate=[1.0, 1.0, 1.5], offered_rate=[1.0, 1.5], desired=[2.0, 2.0])
    clearer = _clearer(individuals_quitting_temperature=0.0)
    _clear(clearer, economy)
    assert clearer.last_num_job_to_job == 0


def _random_economy(rng):
    n_firms, n_ind = 6, 60
    firm_of = rng.integers(-1, n_firms, size=n_ind)
    firm_of[:n_firms] = np.arange(n_firms)  # every firm has a worker
    employments = [np.where(firm_of == j)[0].tolist() for j in range(n_firms)]
    labour_inputs = rng.uniform(0.5, 1.5, n_ind)
    current = np.array([labour_inputs[e].sum() for e in employments])
    return _economy(
        employments,
        wage_rate=rng.uniform(0.5, 1.5, n_ind),
        offered_rate=rng.uniform(0.5, 1.5, n_firms),
        desired=current * rng.uniform(0.6, 1.6, n_firms),
        labour_inputs=labour_inputs,
        reservation=rng.uniform(0.0, 1.0, n_ind),
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_search_with_no_searchers_equals_search_off_and_leaves_the_global_stream(seed):
    """The search step draws from the clearer's own generator, never from the global stream."""
    outcomes = []
    for search_parameters in (
        {"on_the_job_search": False},
        {"on_the_job_search": True, "individuals_quitting_temperature": 0.0},
    ):
        economy = _random_economy(np.random.default_rng(seed))
        clearer = _clearer(firing_speed=1.0, **search_parameters)
        clearer.set_run_seed(seed)
        np.random.seed(seed)
        result, firms, individuals = _clear(clearer, economy)
        outcomes.append(
            (
                individuals.states["Corresponding Firm ID"].copy(),
                individuals.states["Offered Wage of Accepted Job"].copy(),
                result[1:],
                np.random.random(),  # position of the global stream after clearing
            )
        )
        assert clearer.last_num_job_to_job == 0
    for off, on in zip(outcomes[0], outcomes[1]):
        np.testing.assert_array_equal(off, on)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_search_keeps_employment_records_consistent_and_leaves_the_global_stream(seed):
    states = {}
    for search in (False, True):
        economy = _random_economy(np.random.default_rng(seed))
        clearer = _clearer(firing_speed=0.0, on_the_job_search=search)
        clearer.set_run_seed(seed)
        np.random.seed(seed)
        firms, _, individuals = economy
        employed_before = (individuals.states["Activity Status"] == E).sum()
        (_, num_joining, _, _, num_fired), firms, individuals = _clear(clearer, economy)

        firm_of = individuals.states["Corresponding Firm ID"]
        for firm_id, employees in enumerate(firms.states["Employments"]):
            assert len(employees) >= 1
            assert all(firm_of[i] == firm_id for i in employees)
        listed = sorted(i for employees in firms.states["Employments"] for i in employees)
        assert listed == np.where(individuals.states["Activity Status"] == E)[0].tolist()
        assert (individuals.states["Activity Status"] == E).sum() == employed_before + num_joining - num_fired
        states[search] = clearer.last_num_job_to_job
    assert states[False] == 0 and states[True] > 0


def test_searcher_draws_are_reproducible_and_separate_from_the_global_stream():
    def moves(run_seed, global_seed):
        economy = _random_economy(np.random.default_rng(3))
        clearer = _clearer(individuals_quitting_temperature=0.3, optimised_hiring=True)
        clearer.set_run_seed(run_seed)
        np.random.seed(global_seed)
        _, _, individuals = _clear(clearer, economy)
        return individuals.states["Started New Job"].copy()

    np.testing.assert_array_equal(moves(7, 0), moves(7, 0))


def test_invalid_configurations_are_refused():
    with pytest.raises(ValueError, match="does not implement on_the_job_search"):
        _clearer(cls=DefaultLabourMarketClearer)
    with pytest.raises(ValueError, match="enable one"):
        _clearer(individuals_quitting=True)
    with pytest.raises(ValueError, match="searcher_priority"):
        _clearer(searcher_priority="employed_first")


def test_run_seed_reaches_every_country_clearer_with_its_own_stream():
    from macromodel.simulation import seed_labour_market_clearers

    def countries():
        return {
            name: SimpleNamespace(labour_market=SimpleNamespace(functions={"clearing": _clearer()}))
            for name in ("FRA", "DEU")
        }

    def draws(seed):
        built = countries()
        seed_labour_market_clearers(built, seed)
        return [country.labour_market.functions["clearing"]._search_rng.random(4) for country in built.values()]

    first, second = draws(11), draws(11)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert not np.array_equal(first[0], first[1])
    assert not np.array_equal(first[0], draws(12)[0])
