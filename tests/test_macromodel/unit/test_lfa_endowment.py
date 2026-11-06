"""Opening endowment/synchronization checks; no registered real-data experiment."""

import pickle
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

RUN_MODEL = Path(__file__).resolve().parents[3] / "run_model"
if str(RUN_MODEL) not in sys.path:
    sys.path.insert(0, str(RUN_MODEL))

from src.experiments.lfa_endowment import OneTimeLFAEndowment  # noqa: E402
from src.experiments.paired_checkpoint import PairedCheckpoint  # noqa: E402

from macromodel.agents.banks.banks import Banks  # noqa: E402
from macromodel.configurations import CountryConfiguration, SimulationConfiguration  # noqa: E402
from macromodel.simulation import Simulation  # noqa: E402
from macromodel.timeseries import TimeSeries  # noqa: E402
from macromodel.util.get_histogram import get_histogram  # noqa: E402


@pytest.fixture
def model():
    lfa = np.array([100.0, -20.0, 300.0])
    ifa, real, debt = np.array([40.0, 20.0, 50.0]), np.array([80.0, 20.0, 100.0]), np.array([10.0, 20.0, 30.0])
    financial, wealth = lfa + ifa, lfa + ifa + real
    hh = SimpleNamespace(
        ts=TimeSeries(
            income=np.array([10.0, 20.0, 30.0]),
            liquid_financial_assets=lfa,
            total_liquid_financial_assets=np.array([lfa.sum()]),
            illiquid_financial_assets=ifa,
            wealth_financial_assets=financial,
            wealth_real_assets=real,
            wealth=wealth,
            net_wealth=wealth - debt,
            debt=debt,
            wealth_histogram=get_histogram(wealth, 1),
        ),
        states={"corr_individuals": [[0], [1], [2]], "Corresponding Bank ID": np.array([0, 1, 0])},
        functions={
            "consumption": SimpleNamespace(
                idiosyncratic_persistence="initial_state", _household_epsilon=np.array([0.1, 0.2, 0.3])
            )
        },
        initial_states={"untouched": np.array([5.0])},
        compute_income=lambda: np.array([10.0, 20.0, 30.0]),
    )
    # Distinct older rows, without aliasing the mutable current state.
    for key, rows in hh.ts.dicts.items():
        rows.insert(0, np.asarray(rows[-1]).copy())
    banks = SimpleNamespace(
        ts=TimeSeries(
            n_banks=2,
            deposits_from_households=np.array([400.0, -20.0]),
            deposits_from_firms=np.array([30.0, 40.0]),
            total_deposits_from_households=np.array([380.0]),
            total_deposits_from_firms=np.array([70.0]),
            equity=np.array([50.0, 60.0]),
            total_outstanding_loans=np.array([100.0, 200.0]),
            deposits=np.array([380.0, -120.0]),
            liability=np.array([480.0, 220.0]),
            deposits_histogram=get_histogram(np.array([380.0, -120.0]), 1),
            liability_histogram=get_histogram(np.array([480.0, 220.0]), 1),
        )
    )
    banks.compute_deposits = MethodType(Banks.compute_deposits, banks)
    country = SimpleNamespace(
        households=hh,
        banks=banks,
        firms=SimpleNamespace(
            ts=TimeSeries(deposits=np.array([30.0, 40.0])), states={"Corresponding Bank ID": np.array([0, 1])}
        ),
        _staged_household_income_shock=None,
        scale=1,
    )
    return SimpleNamespace(countries={"FRA": country}, prehooks=[], posthooks=[])


def install(model, amounts=(2.0, 30.0, 6.0)):
    gift = OneTimeLFAEndowment("FRA", year=2019, month=1, amounts_nominal=amounts)
    gift.install(model)
    return gift


def test_opening_stock_and_bank_sync_preserve_old_rows_and_other_state(model):
    country = model.countries["FRA"]
    hh, banks = country.households, country.banks
    old_lfa = hh.ts.current("liquid_financial_assets")
    old_rows = {key: pickle.dumps(rows[:-1]) for key, rows in hh.ts.dicts.items()}
    untouched = pickle.dumps((hh.initial_states, hh.ts.current("income"), hh.ts.current("debt"), hh.functions))
    lengths = [{key: len(rows) for key, rows in ts.dicts.items()} for ts in (hh.ts, banks.ts)]
    gift = install(model)
    gift.stage(model, 2019, 1)
    np.testing.assert_array_equal(old_lfa, [100, -20, 300])
    np.testing.assert_array_equal(hh.ts.current("liquid_financial_assets"), [102, 10, 306])
    np.testing.assert_array_equal(banks.ts.current("deposits_from_households"), [408, 10])
    np.testing.assert_array_equal(banks.ts.current("deposits"), [388, -90])
    np.testing.assert_array_equal(banks.ts.current("liability"), [488, 230])
    assert hh.ts.current("total_liquid_financial_assets")[0] == 418
    assert pickle.dumps((hh.initial_states, hh.ts.current("income"), hh.ts.current("debt"), hh.functions)) == untouched
    for key, rows in hh.ts.dicts.items():
        assert pickle.dumps(rows[:-1]) == old_rows[key]
    for ts, counts in zip((hh.ts, banks.ts), lengths):
        assert {k: len(v) for k, v in ts.dicts.items()} == counts
    before = dict(gift.opening_snapshot.household_before)["liquid_financial_assets"]
    np.testing.assert_array_equal(before, old_lfa)
    assert not before.flags.writeable
    gift.stage(model, 2019, 1)
    np.testing.assert_array_equal(hh.ts.current("liquid_financial_assets"), [102, 10, 306])
    assert country._staged_household_income_shock is None


def test_zero_endowment_is_exact_no_op(model):
    gift = install(model, [0, 0, 0])
    country = model.countries["FRA"]
    before = pickle.dumps((country.households.ts.dicts, country.banks.ts.dicts))
    gift.stage(model, 2019, 1)
    assert pickle.dumps((country.households.ts.dicts, country.banks.ts.dicts)) == before


def test_checkpoint_and_control_are_untouched(model):
    checkpoint = PairedCheckpoint(model)
    shock, control = checkpoint.clone(), checkpoint.clone()
    gift = install(shock)
    gift.stage(shock, 2019, 1)
    np.testing.assert_array_equal(
        control.countries["FRA"].households.ts.current("liquid_financial_assets"), [100, -20, 300]
    )
    np.testing.assert_array_equal(
        checkpoint.clone().countries["FRA"].households.ts.current("liquid_financial_assets"), [100, -20, 300]
    )


@pytest.mark.parametrize("fault", ["uninitialized", "residual", "income", "capture", "wealth", "bank", "overflow"])
def test_invalid_opening_state_fails_before_writes(model, fault):
    country = model.countries["FRA"]
    if fault == "uninitialized":
        country.households.ts.income.pop()
    elif fault == "residual":
        country.households.functions["consumption"]._household_epsilon = None
    elif fault == "income":
        country._staged_household_income_shock = np.ones(3)
    elif fault == "capture":
        country._consumption_input_adapter = SimpleNamespace(opening_snapshot=SimpleNamespace(period_id=2))
    elif fault == "wealth":
        country.households.ts.current("wealth")[:] += 100
    elif fault == "bank":
        country.banks.ts.current("deposits")[:] += 1
    elif fault == "overflow":
        # Large finite inputs, aggregate endowment itself finite, sum overflows.
        country.households.ts.override_current("liquid_financial_assets", np.array([1e308, -20.0, 300.0]))
        country.households.ts.override_current("total_liquid_financial_assets", np.array([1e308]))
        for key, vector in {
            "wealth_financial_assets": [1e308, 0.0, 350.0],
            "wealth": [1e308, 20.0, 450.0],
            "net_wealth": [1e308, 0.0, 420.0],
        }.items():
            country.households.ts.override_current(key, np.asarray(vector))
        # Keep the pre-intervention bank accounts coherent, so overflow is
        # reached in the endowment arithmetic rather than baseline validation.
        bank_updates = OneTimeLFAEndowment._bank_values(
            country, country.households.ts.current("liquid_financial_assets")
        )
        for key, value in bank_updates.items():
            country.banks.ts.override_current(key, value)
    gift = install(model, [1e308, 0, 0] if fault == "overflow" else [2, 30, 6])
    before = pickle.dumps((country.households.ts.dicts, country.banks.ts.dicts))
    with pytest.raises(ValueError):
        gift.stage(model, 2019, 1)
    assert pickle.dumps((country.households.ts.dicts, country.banks.ts.dicts)) == before
    assert gift.opening_snapshot is None
    assert gift.booking is None


def test_missed_date_and_unstaged_settlement_rejected(model):
    gift = install(model)
    with pytest.raises(RuntimeError, match="missed"):
        gift.stage(model, 2019, 4)
    with pytest.raises(RuntimeError, match="without successful staging"):
        gift.verify_settlement(model, 20, 2019, 1)


def test_stock_endowment_cannot_be_reported_as_income(model):
    gift = install(model)
    gift.stage(model, 2019, 1)
    hh = model.countries["FRA"].households
    hh.ts.income.append(hh.compute_income() + gift.amounts_nominal)
    with pytest.raises(RuntimeError, match="not be recorded as current income"):
        gift.verify_settlement(model, 20, 2019, 1)
    assert gift.booking is None


def test_native_endowment_settlement_after_unshocked_initialization(datawrapper):
    simulation = Simulation.from_datawrapper(
        datawrapper=datawrapper,
        simulation_configuration=SimulationConfiguration(
            country_configurations={"FRA": CountryConfiguration()}, seed=15
        ),
    )
    simulation.iterate(t=0)  # Synthetic-fixture initialization, not an experiment.
    checkpoint = PairedCheckpoint(simulation)
    arm = checkpoint.clone()
    country = arm.countries["FRA"]
    amounts = np.full(country.households.ts.current("n_households"), 2.5)
    gift = OneTimeLFAEndowment("FRA", year=arm.timestep.year, month=arm.timestep.month, amounts_nominal=amounts)
    gift.install(arm)
    with checkpoint.activate_rng():
        arm.iterate(t=1)
    assert gift.booking is not None
    assert gift.booking.payer_outflow_nominal == float(amounts.sum())
    assert country._staged_household_income_shock is None
    np.testing.assert_array_equal(
        dict(gift.opening_snapshot.household_after)["liquid_financial_assets"]
        - dict(gift.opening_snapshot.household_before)["liquid_financial_assets"],
        amounts,
    )


def test_liability_retains_native_lag_offset(model):
    country = model.countries["FRA"]
    country.banks.ts.current("liability")[:] += 17
    gift = install(model)
    gift.stage(model, 2019, 1)
    # Recomputing from the new balancing position would erase this offset.
    np.testing.assert_array_equal(country.banks.ts.current("liability"), [505, 247])
