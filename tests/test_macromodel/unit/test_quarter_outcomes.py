"""Native outcome capture, availability and pass-through response interoperability."""

import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

RUN_MODEL = Path(__file__).resolve().parents[3] / "run_model"
if str(RUN_MODEL) not in sys.path:
    sys.path.insert(0, str(RUN_MODEL))

from src.experiments.income_transfer import OneTimeIncomeTransfer, TransferBooking  # noqa: E402
from src.experiments.paired_checkpoint import PairedCheckpoint  # noqa: E402
from src.experiments.paired_response import measure_paired_response  # noqa: E402
from src.experiments.quarter_outcomes import QuarterOutcomeCollector  # noqa: E402

from macromodel.configurations import CountryConfiguration, SimulationConfiguration  # noqa: E402
from macromodel.simulation import Simulation  # noqa: E402
from macromodel.timeseries import TimeSeries  # noqa: E402


def model_stub():
    hh = SimpleNamespace(
        ts=TimeSeries(
            income=np.array([10.0, 20.0]),
            consumption=np.array([5.0, 10.0]),
            consumption_cash_expenditure=np.array([7.0, 14.0]),
            consumption_including_housing=np.array([8.0, 16.0]),
            liquid_financial_assets=np.array([100.0, 200.0]),
            cash_saving_before_financing=np.array([np.nan, np.nan]),
        ),
        states={"corr_individuals": [[0], [1]], "Corresponding Bank ID": np.array([0, 0])},
    )
    economy = SimpleNamespace(
        ts=TimeSeries(cpi=np.array([100.0]), gdp_output=np.array([1000.0])),
        current_consumer_price_level=lambda: float(economy.ts.current("cpi")[0]),
        initial_consumer_price_level=lambda: float(economy.ts.initial("cpi")[0]),
        consumer_price_level_series_name=lambda: "cpi",
    )
    country = SimpleNamespace(
        households=hh,
        economy=economy,
        banks=SimpleNamespace(ts=TimeSeries(deposits=np.array([300.0]))),
        central_government=SimpleNamespace(ts=TimeSeries(debt=np.array([500.0]))),
    )
    return SimpleNamespace(countries={"FRA": country}, timestep=SimpleNamespace(year=2019, month=1), posthooks=[])


def collector(model, **changes):
    args = dict(revision="fixture", arm="control", seed=12, start_iteration=0, household_ids={"FRA": ["h0", "h1"]})
    args.update(changes)
    return QuarterOutcomeCollector(model, **args)


def settle(model, *, price=200.0, delta=0.0):
    country = model.countries["FRA"]
    for key in ("income", "consumption", "consumption_cash_expenditure", "consumption_including_housing"):
        ts = country.households.ts
        ts.dicts[key].append(ts.current(key).copy() + delta)
    country.economy.ts.cpi.append(np.array([price]))


def fire(model, t):
    for hook in tuple(model.posthooks):
        hook(model, t, model.timestep.year, model.timestep.month)


def test_native_field_identity_deflator_missingness_and_immutability():
    model = model_stub()
    with collector(model) as capture:
        settle(model)
        fire(model, 0)
        row = capture.records[0]
        assert (row.completed_row, row.iteration, row.year, row.month) == (1, 0, 2019, 1)
        assert row.household_ids == ("h0", "h1") and row.deflator == 2
        np.testing.assert_array_equal(row.nominal_outcomes["consumption_including_housing"], [8, 16])
        assert row.unavailable["funding_plan.credit_granted"] == "missing"
        assert row.unavailable["household.cash_saving_before_financing"] == "nonfinite"
        assert np.isnan(row.household_state["cash_saving_before_financing"]).all()
        model.countries["FRA"].households.ts.current("consumption")[:] = 999
        np.testing.assert_array_equal(row.nominal_outcomes["consumption"], [5, 10])
        with pytest.raises(ValueError):
            row.nominal_outcomes["consumption"].setflags(write=True)
        with pytest.raises(TypeError):
            row.nominal_outcomes["income"] = 0
        assert row.source_rows["outcome.income"] == 2
    assert not model.posthooks


def test_fund_outflow_is_exposed_only_in_booking_quarter():
    model = model_stub()
    source = SimpleNamespace(date=(2019, 1), booking=TransferBooking(2019, 1, 1, np.array([2, 3]), np.array([5]), 5.0))
    with collector(model, interventions={"FRA": source}) as capture:
        settle(model)
        fire(model, 0)
        model.timestep.month = 4
        settle(model)
        fire(model, 1)
    assert capture.records[0].external_funding["payer_outflow_nominal"] == 5
    assert capture.records[1].external_funding["status"] == "no_new_booking"
    assert "payer_outflow_nominal" not in capture.records[1].external_funding


def test_hook_order_is_required_for_due_intervention():
    model = model_stub()
    source = SimpleNamespace(date=(2019, 1), booking=None)
    with collector(model, interventions={"FRA": source}) as capture:
        settle(model)
        with pytest.raises(ValueError, match="after intervention"):
            fire(model, 0)
        assert not capture.records


@pytest.mark.parametrize(
    "fault", ["nan", "shape", "stale_outcome", "stale_price", "price", "reorder", "duplicate", "date"]
)
def test_invalid_completed_state_never_records_partial_snapshot(fault):
    model = model_stub()
    with collector(model) as capture:
        settle(model)
        country = model.countries["FRA"]
        if fault == "nan":
            country.households.ts.current("income")[:] = np.nan
        elif fault == "shape":
            country.households.ts.override_current("consumption", np.array([1.0]))
        elif fault == "stale_outcome":
            country.households.ts.consumption.pop()
        elif fault == "stale_price":
            country.economy.ts.cpi.pop()
        elif fault == "price":
            country.economy.ts.override_current("cpi", [0.0])
        elif fault == "reorder":
            country.households.states["corr_individuals"].reverse()
        elif fault == "duplicate":
            fire(model, 0)
            with pytest.raises(ValueError):
                fire(model, 0)
            assert len(capture.records) == 1
            return
        elif fault == "date":
            with pytest.raises(ValueError):
                capture._capture(model, 0, 2019, 4)
            assert not capture.records
            return
        with pytest.raises(ValueError):
            fire(model, 0)
        assert not capture.records


def test_missing_native_outcome_remains_explicit_and_hooks_restore_on_exception():
    model = model_stub()

    def existing(*_):
        return None

    model.posthooks.append(existing)
    with pytest.raises(RuntimeError):
        with collector(model) as capture:
            del model.countries["FRA"].households.ts.dicts["consumption_including_housing"]
            country = model.countries["FRA"]
            for key in ("income", "consumption", "consumption_cash_expenditure"):
                country.households.ts.dicts[key].append(country.households.ts.current(key).copy())
            country.economy.ts.cpi.append([200.0])
            fire(model, 0)
            assert capture.records[0].unavailable["outcome.consumption_including_housing"] == "missing"
            raise RuntimeError("fixture error")
    assert model.posthooks == [existing]


def test_collected_nominal_paths_feed_response_measurement():
    paths = []
    for price, delta in ((200.0, 0.0), (400.0, 20.0)):
        model = model_stub()
        with collector(model) as capture:
            settle(model, price=price, delta=delta)
            fire(model, 0)
        paths.append(capture.records[0])
    a, b = paths
    result = measure_paired_response(
        outcome="consumption",
        intervention_kind="income_transfer",
        value_units="nominal",
        control_nominal=[a.nominal_outcomes["consumption"]],
        shock_nominal=[b.nominal_outcomes["consumption"]],
        control_ids=a.household_ids,
        shock_ids=b.household_ids,
        control_periods=[a.completed_row],
        shock_periods=[b.completed_row],
        control_deflators=[a.deflator],
        shock_deflators=[b.deflator],
        intervention_nominal=[10, 10],
        pre_shock_control_deflator=1.0,
        horizons=(1,),
    )
    # (25/4 - 5/2 + 30/4 - 10/2)/20 = 6.25/20.
    assert result.summaries[1]["all"]["real_response"] == 0.3125


def test_native_settlement_collector_is_transparent_and_after_transfer(datawrapper):
    model = Simulation.from_datawrapper(
        datawrapper=datawrapper,
        simulation_configuration=SimulationConfiguration(
            country_configurations={"FRA": CountryConfiguration()}, seed=15
        ),
    )
    checkpoint = PairedCheckpoint(model)
    observed, control = checkpoint.clone(), checkpoint.clone()
    h = observed.countries["FRA"].households.ts.current("n_households")
    sources = []
    for arm in (observed, control):
        source = OneTimeIncomeTransfer(
            "FRA", year=arm.timestep.year, month=arm.timestep.month, amounts_nominal=np.full(h, 2.5)
        )
        source.install(arm)
        sources.append(source)
    rng_before = pickle.dumps(np.random.get_state())
    with checkpoint.activate_rng():
        control.iterate(t=0)
    with checkpoint.activate_rng():
        with collector(
            observed, seed=15, household_ids={"FRA": list(range(h))}, interventions={"FRA": sources[0]}
        ) as capture:
            observed.iterate(t=0)
    assert pickle.dumps(np.random.get_state()) == rng_before
    hh, baseline = observed.countries["FRA"].households, control.countries["FRA"].households
    for key in hh.ts.dicts:
        np.testing.assert_array_equal(np.asarray(hh.ts.historic(key)), np.asarray(baseline.ts.historic(key)))
    row = capture.records[0]
    assert row.external_funding["payer_outflow_nominal"] == h * 2.5
    assert row.deflator > 0 and not any(key.startswith("outcome.") for key in row.unavailable)
    np.testing.assert_array_equal(
        row.nominal_outcomes["consumption_including_housing"], hh.ts.current("consumption_including_housing")
    )
