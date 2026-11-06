"""Checkpoint isolation and seeded replay without running macro experiments."""

import pickle
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from numba import njit

RUN_MODEL = Path(__file__).resolve().parents[3] / "run_model"
if str(RUN_MODEL) not in sys.path:
    sys.path.insert(0, str(RUN_MODEL))

from src.experiments.paired_checkpoint import PairedCheckpoint, _RandomState  # noqa: E402

from macromodel.configurations import CountryConfiguration, SimulationConfiguration  # noqa: E402
from macromodel.simulation import Simulation  # noqa: E402
from macromodel.timeseries import TimeSeries  # noqa: E402


@njit
def compiled_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


@njit
def compiled_draws():
    return np.random.normal(), random.gauss(0.0, 1.0), np.random.random(), random.random()


@pytest.fixture(autouse=True)
def preserve_rng():
    state = _RandomState.capture()
    try:
        yield
    finally:
        state.restore()


def model_stub():
    ts = TimeSeries(balance=np.array([10.0, 20.0]))
    owner = SimpleNamespace(ts=ts, arrays={"stock": np.array([3.0])})
    owner.self = owner
    return SimpleNamespace(
        prehooks=[],
        posthooks=[],
        countries={"FRA": owner},
        shared=ts,
        nested={"ts": [ts]},
        rng=np.random.default_rng(123),
        legacy_rng=np.random.RandomState(456),
        timestep=SimpleNamespace(month=1),
    )


def test_clone_preserves_aliases_cycles_and_independent_arrays():
    model = model_stub()
    checkpoint = PairedCheckpoint(model)
    model.shared.current("balance")[:] = -1
    a, b = checkpoint.clone(), checkpoint.clone()
    assert a.shared is a.countries["FRA"].ts is a.nested["ts"][0]
    assert a.countries["FRA"].self is a.countries["FRA"]
    np.testing.assert_array_equal(a.shared.current("balance"), [10, 20])
    a.shared.current("balance")[:] = 99
    a.countries["FRA"].arrays["stock"][:] = 40
    a.timestep.month = 8
    np.testing.assert_array_equal(b.shared.current("balance"), [10, 20])
    np.testing.assert_array_equal(b.countries["FRA"].arrays["stock"], [3])
    assert b.timestep.month == 1
    np.testing.assert_array_equal(a.rng.normal(size=10), b.rng.normal(size=10))
    np.testing.assert_array_equal(a.legacy_rng.normal(size=10), b.legacy_rng.normal(size=10))
    a.rng.normal(size=2)
    assert pickle.dumps(a.rng.bit_generator.state) != pickle.dumps(b.rng.bit_generator.state)


def test_checkpoint_and_clone_do_not_consume_rng_and_replay_cached_normals():
    random.seed(71)
    np.random.seed(72)
    compiled_seed(73)
    # One normal draw leaves the second Box-Muller draw cached in each stream.
    compiled_draws()
    before = pickle.dumps(_RandomState.capture())
    checkpoint = PairedCheckpoint(model_stub())
    checkpoint.clone()
    assert pickle.dumps(_RandomState.capture()) == before
    results = []
    for _ in range(2):
        with checkpoint.activate_rng():
            results.append((random.gauss(0, 1), np.random.normal(size=4), compiled_draws()))
        assert pickle.dumps(_RandomState.capture()) == before
    assert results[0][0] == results[1][0]
    np.testing.assert_array_equal(results[0][1], results[1][1])
    assert results[0][2] == results[1][2]


def test_exception_restores_ambient_streams():
    checkpoint = PairedCheckpoint(model_stub())
    np.random.random(11)
    random.random()
    compiled_draws()
    ambient = pickle.dumps(_RandomState.capture())
    with pytest.raises(RuntimeError, match="arm failed"):
        with checkpoint.activate_rng():
            np.random.random(5)
            random.random()
            compiled_draws()
            raise RuntimeError("arm failed")
    assert pickle.dumps(_RandomState.capture()) == ambient


@pytest.mark.parametrize("hook_field", ["prehooks", "posthooks"])
def test_existing_hooks_rejected(hook_field):
    model = model_stub()
    getattr(model, hook_field).append(lambda: model)
    with pytest.raises(ValueError, match="before installing hooks"):
        PairedCheckpoint(model)


def test_cross_thread_use_rejected():
    checkpoint = PairedCheckpoint(model_stub())
    with ThreadPoolExecutor(max_workers=1) as executor:
        with pytest.raises(RuntimeError, match="original process and thread"):
            executor.submit(checkpoint.clone).result()


@pytest.mark.parametrize("legacy_copy", [False, True])
def test_complete_simulation_copy_without_advancing(datawrapper, monkeypatch, legacy_copy):
    model = Simulation.from_datawrapper(
        datawrapper=datawrapper,
        simulation_configuration=SimulationConfiguration(
            country_configurations={"FRA": CountryConfiguration()},
            seed=15,
        ),
    )
    country = model.countries["FRA"]
    if legacy_copy:
        # Reproduce old-main's missing Country copy override, without backporting
        # integrated consumption logic or executing an archived revision.
        monkeypatch.delattr(type(country), "__deepcopy__")
    rule = country.households.functions["consumption"]
    rule._household_epsilon = np.arange(country.households.ts.current("n_households"), dtype=float)
    checkpoint = PairedCheckpoint(model)
    a, b = checkpoint.clone(), checkpoint.clone()
    assert a.goods_market is not b.goods_market
    assert a.countries["FRA"]._consumption_input_adapter is not b.countries["FRA"]._consumption_input_adapter
    assert a.rest_of_the_world is not b.rest_of_the_world
    assert a.exchange_rates is not b.exchange_rates
    assert a.timestep is not b.timestep
    first, second = a.countries["FRA"], b.countries["FRA"]
    np.testing.assert_array_equal(first.households.functions["consumption"]._household_epsilon, rule._household_epsilon)
    first.households.functions["consumption"]._household_epsilon[:] = -100
    np.testing.assert_array_equal(
        second.households.functions["consumption"]._household_epsilon, rule._household_epsilon
    )
    opening = second.households.ts.current("liquid_financial_assets").copy()
    first.households.ts.current("liquid_financial_assets")[:] += 100
    np.testing.assert_array_equal(second.households.ts.current("liquid_financial_assets"), opening)
    # Synthetic random consumers verify zero-intervention continuation state;
    # no model iteration or production experiment occurs in this task.
    values = []
    for arm in (a, b):
        with checkpoint.activate_rng():
            values.append((np.random.random(), compiled_draws()))
            arm.timestep.step()
    assert values[0] == values[1]
    assert a.timestep.year == b.timestep.year
    assert a.timestep.month == b.timestep.month


def test_unsupported_numba_layout_fails_without_changing_python_numpy(monkeypatch):
    from src.experiments import paired_checkpoint

    python_state = random.getstate()
    numpy_state = pickle.dumps(np.random.get_state())
    with monkeypatch.context() as patch:
        patch.setattr(paired_checkpoint.randomimpl, "rnd_state_t", "unsupported")
        with pytest.raises(RuntimeError, match="Unsupported Numba RNG layout"):
            PairedCheckpoint(model_stub())
    assert random.getstate() == python_state
    assert pickle.dumps(np.random.get_state()) == numpy_state
