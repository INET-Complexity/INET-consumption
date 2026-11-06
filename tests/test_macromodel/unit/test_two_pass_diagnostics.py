"""Pass snapshots, observer transparency and native integrated target coverage."""

import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

RUN_MODEL = Path(__file__).resolve().parents[3] / "run_model"
if str(RUN_MODEL) not in sys.path:
    sys.path.insert(0, str(RUN_MODEL))

from src.experiments.paired_checkpoint import PairedCheckpoint  # noqa: E402
from src.experiments.two_pass_diagnostics import TwoPassDiagnosticCapture  # noqa: E402

from macromodel.configurations import (  # noqa: E402
    CountryConfiguration,
    SimulationConfiguration,
    load_country_configuration,
)
from macromodel.simulation import Simulation  # noqa: E402
from macromodel.timeseries import TimeSeries  # noqa: E402


class DummyHouseholds:
    def __init__(self):
        self.ts = TimeSeries(income=np.array([10.0, 20.0]), liquid_financial_assets=np.array([100.0, 200.0]))
        self.functions = {"consumption": SimpleNamespace(last_target_consumption_components={})}
        self.calls = 0

    def compute_target_consumption(self, amount=1.0, replace_current_diagnostics=False, assume_zero_growth=False):
        self.calls += 1
        result = np.array([[amount], [amount * 2]])
        self.functions["consumption"].last_target_consumption_components = {
            "target_consumption_raw_target_real": result[:, 0],
        }
        return result


def model_stub():
    return SimpleNamespace(
        countries={"FRA": SimpleNamespace(households=DummyHouseholds())},
        timestep=SimpleNamespace(year=2019, month=1),
        prehooks=[],
        posthooks=[],
    )


def test_both_passes_are_immutable_independent_and_native_called_once():
    model = model_stub()
    hh = model.countries["FRA"].households
    with TwoPassDiagnosticCapture(model) as capture:
        first = hh.compute_target_consumption(2.0)
        hh.ts.current("liquid_financial_assets")[:] = 3
        second = hh.compute_target_consumption(amount=5.0, replace_current_diagnostics=True)
        first[:] = -10
        second[:] = -20
    assert hh.calls == 2
    assert [row.pass_kind for row in capture.records] == ["first", "replacement"]
    a, b = capture.records
    assert (a.country_code, a.period_id, a.year, a.month) == ("FRA", 1, 2019, 1)
    np.testing.assert_array_equal(a.goods_demand, [[2], [4]])
    np.testing.assert_array_equal(b.goods_demand, [[5], [10]])
    np.testing.assert_array_equal(a.household_before["liquid_financial_assets"]["current"], [100, 200])
    np.testing.assert_array_equal(a.components["target_consumption_raw_target_real"], [2, 4])
    with pytest.raises(ValueError):
        a.goods_demand.setflags(write=True)
    with pytest.raises(TypeError):
        a.inputs["amount"] = 8
    assert "compute_target_consumption" not in vars(hh)


def test_missing_fields_and_bypass_do_not_reuse_stale_components():
    model = model_stub()
    hh = model.countries["FRA"].households
    with TwoPassDiagnosticCapture(model) as capture:
        hh.compute_target_consumption()
        hh.compute_target_consumption(assume_zero_growth=True)
    assert "target_consumption_income_base" in capture.records[0].unavailable_components
    assert not capture.records[1].components
    assert "target_consumption_raw_target_real" in capture.records[1].unavailable_components
    assert "target_consumption_income_base" not in capture.records[0].components


def test_failure_restores_existing_instance_override_and_records_no_success():
    model = model_stub()
    hh = model.countries["FRA"].households

    def fail(amount=1.0):
        raise RuntimeError("native failure")

    hh.compute_target_consumption = fail
    with pytest.raises(RuntimeError, match="native failure"):
        with TwoPassDiagnosticCapture(model) as capture:
            hh.compute_target_consumption()
    assert hh.compute_target_consumption is fail
    assert capture.records == ()


def test_partial_install_failure_restores_earlier_country():
    model = model_stub()
    model.countries["BAD"] = SimpleNamespace(households=SimpleNamespace())
    with pytest.raises(AttributeError):
        with TwoPassDiagnosticCapture(model):
            pass
    assert "compute_target_consumption" not in vars(model.countries["FRA"].households)


def test_reenter_rejected_and_exception_restores_class_method():
    model = model_stub()
    capture = TwoPassDiagnosticCapture(model)
    with capture:
        with pytest.raises(RuntimeError, match="more than once"):
            capture.__enter__()
    with pytest.raises(RuntimeError, match="more than once"):
        capture.__enter__()
    assert "compute_target_consumption" not in vars(model.countries["FRA"].households)


@pytest.fixture
def integrated_model(datawrapper, monkeypatch):
    root = RUN_MODEL.parent
    production = load_country_configuration(root / "run_model/config/country_config_FRA.yaml", country_iso3="FRA")
    config = CountryConfiguration()
    config.households = production.households
    config.households.functions.wealth.parameters["frm_coefficients_path"] = str(
        root / "run_model/data/raw_data/portfolio/FR_portfolio_frm_coefficients.json"
    )
    config.banks.parameters.household_consumption_loan_maturity = (
        production.banks.parameters.household_consumption_loan_maturity
    )
    config.banks.parameters.requires_household_underwriting_snapshot = True
    config.consumer_credit = production.consumer_credit
    config.credit_market = production.credit_market
    simulation = Simulation.from_datawrapper(
        datawrapper=datawrapper,
        simulation_configuration=SimulationConfiguration(country_configurations={"FRA": config}, seed=15),
    )

    # Direct seam calls intentionally skip estimation; give both copied arms
    # the same finite expected inflation, as in existing adapter integration tests.
    monkeypatch.setattr(simulation.countries["FRA"].economy, "current_expected_consumer_period_inflation", lambda: 0.0)
    return simulation


def test_native_two_pass_capture_is_transparent_and_retains_prepared_inputs(integrated_model):
    checkpoint = PairedCheckpoint(integrated_model)
    observed, control = checkpoint.clone(), checkpoint.clone()

    def two_passes(arm):
        country = arm.countries["FRA"]
        country._set_household_target_demand(replace_current=False)
        hh = country.households
        hh.ts.override_current("expected_income_employee", hh.ts.current("expected_income_employee") * 1.5)
        country._set_household_target_demand(replace_current=True)

    before = pickle.dumps(np.random.get_state())
    with checkpoint.activate_rng():
        two_passes(control)
    with checkpoint.activate_rng():
        with TwoPassDiagnosticCapture(observed) as capture:
            two_passes(observed)
    assert pickle.dumps(np.random.get_state()) == before
    hh = observed.countries["FRA"].households
    baseline = control.countries["FRA"].households
    for key in hh.ts.dicts:
        np.testing.assert_array_equal(np.asarray(hh.ts.historic(key)), np.asarray(baseline.ts.historic(key)))
    a, b = capture.records
    assert a.pass_kind == "first" and b.pass_kind == "replacement"
    assert a.period_id == b.period_id
    prepared = a.inputs["consumption_inputs"]
    np.testing.assert_array_equal(a.components["target_consumption_income_base"], prepared["income_base_real"])
    np.testing.assert_array_equal(
        prepared["opening_real"]["liquid_financial_assets"],
        b.inputs["consumption_inputs"]["opening_real"]["liquid_financial_assets"],
    )
    assert not a.unavailable_components
    assert np.any(a.components["target_consumption_income_base"] != b.components["target_consumption_income_base"])
    # The first pass survives replacement and later live-state mutation.
    saved = a.components["target_consumption_income_base"].copy()
    hh.functions["consumption"].last_target_consumption_components["target_consumption_income_base"] = np.zeros_like(
        saved
    )
    np.testing.assert_array_equal(a.components["target_consumption_income_base"], saved)
