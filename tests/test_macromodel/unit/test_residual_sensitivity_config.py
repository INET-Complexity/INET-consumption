"""Synthetic config, startup pairing and explicit-draw country reset checks."""

import pickle
import random
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from numba import njit

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "run_model"))
from src.experiments.paired_checkpoint import _RandomState  # noqa: E402
from src.experiments.residual_sensitivity_config import (  # noqa: E402
    ResidualCase,
    proposed_residual_cases,
    residual_case_configuration,
)

from macromodel.configurations import (  # noqa: E402
    CountryConfiguration,
    SimulationConfiguration,
    load_country_configuration,
)
from macromodel.simulation import Simulation  # noqa: E402


@njit
def seed_compiled(seed):
    np.random.seed(seed)
    random.seed(seed)


@pytest.fixture(autouse=True)
def preserve_rng():
    state = _RandomState.capture()
    yield
    state.restore()


@pytest.fixture
def base():
    production = load_country_configuration(ROOT / "run_model/config/country_config_FRA.yaml", "FRA")
    country = CountryConfiguration()
    country.households = deepcopy(production.households)
    country.households.functions.wealth.parameters["frm_coefficients_path"] = str(
        ROOT / "run_model/data/raw_data/portfolio/FR_portfolio_frm_coefficients.json"
    )
    country.banks.parameters.household_consumption_loan_maturity = (
        production.banks.parameters.household_consumption_loan_maturity
    )
    country.banks.parameters.requires_household_underwriting_snapshot = True
    country.consumer_credit = deepcopy(production.consumer_credit)
    country.credit_market = deepcopy(production.credit_market)
    return SimulationConfiguration(country_configurations={"FRA": country}, seed=12)


def test_matrix_is_crossed_and_anchored_references_are_unique():
    cases = proposed_residual_cases()
    assert len(cases) == len(set(cases)) == 12
    assert sum(c.policy == "anchored" for c in cases) == 3
    assert {(c.model_seed, c.draw_seed) for c in cases if c.policy == "drawn"} == {
        (m, d) for m in (12, 13, 14) for d in (12001, 12002, 12003)
    }


@pytest.mark.parametrize("case", proposed_residual_cases())
def test_overlay_changes_only_declared_values_and_owns_nested_config(base, case):
    before = base.model_dump()
    config = residual_case_configuration(base, case)
    expected = deepcopy(before)
    expected["seed"] = case.model_seed
    params = expected["country_configurations"]["FRA"]["households"]["functions"]["consumption"]["parameters"]
    if case.policy == "drawn":
        params.update(idiosyncratic_persistence="fixed_effect", idiosyncratic_seed=case.draw_seed)
    assert config.model_dump() == expected and base.model_dump() == before
    config.country_configurations["FRA"].households.functions.consumption.parameters["long_run_intercept"] = 999
    assert base.model_dump() == before


@pytest.mark.parametrize(
    "case",
    [
        ResidualCase(True, "anchored"),
        ResidualCase(-1, "anchored"),
        ResidualCase(12, "bad"),
        ResidualCase(12, "drawn"),
        ResidualCase(12, "anchored", 1),
        ResidualCase(12, "drawn", True),
        ResidualCase(12, "drawn", 2**32),
    ],
)
def test_bad_case_rejected(base, case):
    with pytest.raises(ValueError):
        residual_case_configuration(base, case)


@pytest.mark.parametrize(
    "key,value",
    [
        ("idiosyncratic_persistence", "fixed_effect"),
        ("idiosyncratic_seed", 7),
        ("idiosyncratic_sd", float("nan")),
        ("idiosyncratic_sd", 0),
    ],
)
def test_bad_baseline_rejected(base, key, value):
    base.country_configurations["FRA"].households.functions.consumption.parameters[key] = value
    with pytest.raises(ValueError):
        residual_case_configuration(base, ResidualCase(12, "drawn", 12001))


def equal_tree(left, right):
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal_tree(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            equal_tree(a, b)
    elif hasattr(left, "equals"):
        assert left.equals(right)
    elif hasattr(left, "__dict__"):
        assert type(left) is type(right)
        equal_tree(vars(left), vars(right))
    else:
        np.testing.assert_equal(left, right)


def economic_snapshot(country):
    result = {}
    for name in (
        "households",
        "individuals",
        "firms",
        "banks",
        "central_government",
        "central_bank",
        "government_entities",
        "economy",
    ):
        agent = getattr(country, name)
        result[name] = deepcopy(agent.ts.dicts)
    result["household_states"] = deepcopy(country.households.states)
    return result


@pytest.mark.parametrize("model_seed", (12, 13, 14))
def test_synthetic_construction_has_common_pre_residual_inputs(base, datawrapper, model_seed):
    random.seed(model_seed)
    np.random.seed(model_seed)
    seed_compiled(model_seed)
    common = _RandomState.capture()
    reference = None
    final_rng = None
    for case in [c for c in proposed_residual_cases() if c.model_seed == model_seed]:
        common.restore()
        config = residual_case_configuration(base, case)
        model = Simulation.from_datawrapper(deepcopy(datawrapper), config)
        country = model.countries["FRA"]
        rule = country.households.functions["consumption"]
        assert rule._household_epsilon is None
        assert getattr(rule, "_prepared_residual_layout", None) is None
        snapshot = economic_snapshot(country)
        state = pickle.dumps(_RandomState.capture())
        if reference is None:
            reference, final_rng = snapshot, state
        else:
            equal_tree(reference, snapshot)
            assert state == final_rng


@pytest.mark.parametrize("draw_seed", (12001, 12002, 12003))
def test_explicit_draw_country_reset_rebuilds_and_reproduces(base, datawrapper, monkeypatch, draw_seed):
    config = residual_case_configuration(base, ResidualCase(12, "drawn", draw_seed))
    model = Simulation.from_datawrapper(deepcopy(datawrapper), config)
    country = model.countries["FRA"]
    monkeypatch.setattr(country.economy, "current_expected_consumer_period_inflation", lambda: 0.0)
    old = country.households.functions["consumption"]
    country._set_household_target_demand(replace_current=False)
    expected = old._household_epsilon.copy()
    np.testing.assert_array_equal(
        expected, np.random.default_rng(draw_seed).normal(0, old.idiosyncratic_sd, len(expected))
    )
    model.reset()
    new = country.households.functions["consumption"]
    assert new is not old and new._household_epsilon is None
    country._set_household_target_demand(replace_current=False)
    np.testing.assert_array_equal(new._household_epsilon, expected)
