"""Per-case runner assembly and real native synthetic-fixture artifact roundtrip."""

import json
import pickle
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

RUN_MODEL = Path(__file__).resolve().parents[3] / "run_model"
if str(RUN_MODEL) not in sys.path:
    sys.path.insert(0, str(RUN_MODEL))

from src.experiments.paired_checkpoint import _RandomState  # noqa: E402
from src.experiments.paired_validation import load_artifact, run_paired_case, save_artifact  # noqa: E402

from macromodel.agents.banks.banks import Banks  # noqa: E402
from macromodel.configurations import CountryConfiguration, SimulationConfiguration  # noqa: E402
from macromodel.country.country import Country  # noqa: E402
from macromodel.simulation import Simulation  # noqa: E402
from macromodel.timeseries import TimeSeries  # noqa: E402


@pytest.fixture(autouse=True)
def preserve_rng():
    state = _RandomState.capture()
    try:
        yield
    finally:
        state.restore()


@pytest.fixture
def output():
    with tempfile.TemporaryDirectory(dir="/private/tmp", prefix="paired-runner-test-") as directory:
        yield Path(directory) / "case"


class ToyHouseholds:
    def __init__(self):
        self.states = {"corr_individuals": [[0], [1]], "Corresponding Bank ID": np.array([0, 1])}
        self.functions = {
            "consumption": SimpleNamespace(idiosyncratic_persistence="none", last_target_consumption_components={})
        }
        self.ts = TimeSeries(
            income=np.full(2, 10.0),
            expected_income=np.full(2, 10.0),
            consumption=np.full(2, 5.0),
            consumption_cash_expenditure=np.full(2, 5.0),
            consumption_including_housing=np.full(2, 5.0),
            liquid_financial_assets=np.array([100.0, 200.0]),
            total_liquid_financial_assets=[300.0],
            illiquid_financial_assets=np.array([20.0, 40.0]),
            wealth_financial_assets=np.array([120.0, 240.0]),
            wealth_real_assets=np.zeros(2),
            wealth=np.array([120.0, 240.0]),
            net_wealth=np.array([120.0, 240.0]),
            debt=np.zeros(2),
        )

    def compute_income(self):
        return np.full(2, 10.0)

    def compute_target_consumption(self, replace_current_diagnostics=False):
        shock = self.country._staged_household_income_shock
        income = self.compute_income() + (0 if shock is None else shock)
        value = income + 0.01 * self.ts.current("liquid_financial_assets") + np.random.random(2)
        if self.country.zero_diverges and shock is not None and not np.any(shock):
            value += 1
        self.functions["consumption"].last_target_consumption_components = {
            "target_consumption_calibrated_total": value,
            "target_consumption_long_run": value * 1.1,
        }
        return value[:, None]


class ToyConfiguration:
    def model_dump(self, **_kwargs):
        return {"fixture": True}


class ToyCountry:
    stage_household_income_shock = Country.stage_household_income_shock
    _apply_household_income_shock = Country._apply_household_income_shock
    _clear_household_income_shock = Country._clear_household_income_shock

    def __init__(self, *, fail=False, zero_diverges=False):
        self.households = ToyHouseholds()
        self.households.country = self
        self.firms = SimpleNamespace(
            ts=TimeSeries(deposits=np.zeros(2)), states={"Corresponding Bank ID": np.array([0, 1])}
        )
        self.banks = ToyBanks()
        self.central_government = SimpleNamespace(ts=TimeSeries(debt=np.array([100.0])))
        self.economy = ToyEconomy()
        self.scale, self.fail, self.zero_diverges = 1, fail, zero_diverges
        self._staged_household_income_shock = None


class ToyBanks:
    compute_deposits = Banks.compute_deposits
    compute_liability = Banks.compute_liability
    update_deposits = Banks.update_deposits

    def __init__(self):
        self.ts = TimeSeries(
            n_banks=2,
            equity=np.zeros(2),
            deposits_from_firms=np.zeros(2),
            total_deposits_from_firms=[0.0],
            deposits_from_households=np.array([100.0, 200.0]),
            total_deposits_from_households=[300.0],
            total_outstanding_loans=np.zeros(2),
            deposits=np.array([100.0, 200.0]),
            liability=np.array([100.0, 200.0]),
        )


class ToyEconomy:
    def __init__(self):
        self.ts = TimeSeries(cpi=np.array([1.0]), gdp_output=np.array([1000.0]))

    def current_consumer_price_level(self):
        return self.ts.current("cpi")[0]

    def initial_consumer_price_level(self):
        return self.ts.initial("cpi")[0]

    def consumer_price_level_series_name(self):
        return "cpi"


class ToyModel:
    def __init__(self, **kwargs):
        self.countries = {"FRA": ToyCountry(**kwargs)}
        self.prehooks, self.posthooks, self.iterations = [], [], []
        self.timestep = SimpleNamespace(year=2019, month=1)
        self.configuration, self.random_seed = ToyConfiguration(), 12

    def iterate(self, t):
        for hook in self.prehooks:
            hook(self, self.timestep.year, self.timestep.month)
        country = self.countries["FRA"]
        hh, banks = country.households, country.banks
        hh.compute_target_consumption()
        if (
            country.fail
            and country._staged_household_income_shock is not None
            and np.any(country._staged_household_income_shock)
        ):
            raise RuntimeError("injected fixture failure")
        spending = hh.compute_target_consumption(replace_current_diagnostics=True).sum(axis=1)
        income = country._apply_household_income_shock(hh.compute_income())
        hh.ts.income.append(income)
        country._clear_household_income_shock()
        for key in ("consumption", "consumption_cash_expenditure", "consumption_including_housing"):
            hh.ts.dicts[key].append(spending.copy())
        lfa = hh.ts.current("liquid_financial_assets") + income - spending
        for key, value in {
            "liquid_financial_assets": lfa,
            "total_liquid_financial_assets": [lfa.sum()],
            "wealth_financial_assets": lfa + hh.ts.current("illiquid_financial_assets"),
            "wealth": lfa + hh.ts.current("illiquid_financial_assets"),
            "net_wealth": lfa + hh.ts.current("illiquid_financial_assets"),
        }.items():
            hh.ts.dicts[key].append(value)
        banks.update_deposits(
            current_firm_deposits=np.zeros(2),
            current_household_deposits=lfa,
            firm_corresponding_bank=np.array([0, 1]),
            households_corresponding_bank=np.array([0, 1]),
        )
        banks.ts.liability.append(banks.compute_liability())
        banks.ts.deposits.append(banks.compute_deposits())
        country.economy.ts.cpi.append(np.array([1.0]))
        self.iterations.append(t)
        for hook in self.posthooks:
            hook(self, t, self.timestep.year, self.timestep.month)
        self.timestep.month += 3
        if self.timestep.month > 12:
            self.timestep.year += 1
            self.timestep.month -= 12


def args(output, **changes):
    values = dict(
        country_code="FRA",
        revision="fixture",
        seed=12,
        start_iteration=1,
        horizon_periods=1,
        household_ids=["a", "b"],
        transfer_nominal=[2.0, 3.0],
        endowment_nominal=[4.0, 5.0],
        horizons=(1,),
        include_zero=True,
        output_dir=output,
        provenance={"source_snapshot": "synthetic fixture, not a production pin"},
    )
    values.update(changes)
    return values


def test_runner_assembles_arms_saves_roundtrip_and_leaves_source_rng_untouched(output):
    model = ToyModel()
    before = pickle.dumps(_RandomState.capture())
    manifest = run_paired_case(model, **args(output))
    assert manifest["status"] == "complete" and manifest["zero_control"] == "passed"
    assert manifest["completed_arms"] == ["control", "zero", "income_transfer", "lfa_endowment"]
    assert model.iterations == [] and not model.prehooks and not model.posthooks
    assert pickle.dumps(_RandomState.capture()) == before
    stock = load_artifact(output / "lfa_endowment")
    assert stock["booking"]["payer_outflow_nominal"] == 9
    assert stock["opening_endowment_snapshot"] is not None
    assert len(stock["passes"]) == 2 and len(stock["quarters"]) == 1
    results = load_artifact(output / "responses")
    assert results["income_transfer"]["consumption"]["response_kind"] == "income_mpc"
    assert results["lfa_endowment"]["replacement/goods_demand"]["response_kind"] == "lfa_spending_response"
    assert manifest["artifact_sha256"]
    with pytest.raises(FileExistsError):
        run_paired_case(model, **args(output))


@pytest.mark.parametrize("fault", ["arm", "zero", "budget"])
def test_failures_preserve_manifest_and_available_arm_evidence(output, fault):
    model = ToyModel(fail=fault == "arm", zero_diverges=fault == "zero")
    kwargs = args(output, wall_seconds=1e-20) if fault == "budget" else args(output)
    with pytest.raises((RuntimeError, ValueError)):
        run_paired_case(model, **kwargs)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed" and (output / "failure.txt").exists()
    if fault == "arm":
        partial = load_artifact(output / "income_transfer")
        assert partial["status"] == "failed" and len(partial["passes"]) == 1
        assert not partial["quarters"] and not (output / "lfa_endowment").exists()
    if fault == "zero":
        assert (output / "zero/records.json").exists()
        assert "income_transfer" not in manifest["completed_arms"]


@pytest.mark.parametrize(
    "changes",
    [
        {"start_iteration": 0},
        {"horizons": (4,)},
        {"household_ids": ["a"]},
        {"transfer_nominal": [np.nan, 1.0]},
        {"provenance": {}},
    ],
)
def test_invalid_case_fails_before_output(output, changes):
    with pytest.raises(ValueError):
        run_paired_case(ToyModel(), **args(output, **changes))
    assert not output.exists()


def test_artifact_preserves_nan_bytes_and_forbids_pickle_objects(output):
    save_artifact(output, {"vector": np.array([1.0, np.nan]), "scalar": float("nan"), "rng": b"\x00\x01"})
    data = load_artifact(output)
    assert np.isnan(data["vector"][1]) and np.isnan(data["scalar"]) and data["rng"] == b"\x00\x01"
    assert "NaN" not in (output / "records.json").read_text()
    with pytest.raises(TypeError):
        save_artifact(output / "bad", np.array([object()], dtype=object))


def test_native_synthetic_fixture_full_runner_case(datawrapper, output):
    model = Simulation.from_datawrapper(
        datawrapper=datawrapper,
        simulation_configuration=SimulationConfiguration(
            country_configurations={"FRA": CountryConfiguration()}, seed=15
        ),
    )
    h = model.countries["FRA"].households.ts.current("n_households")
    manifest = run_paired_case(
        model,
        **args(
            output,
            seed=15,
            household_ids=list(range(h)),
            transfer_nominal=np.full(h, 2.5),
            endowment_nominal=np.full(h, 3.0),
        ),
    )
    assert manifest["status"] == "complete" and manifest["zero_control"] == "passed"
    assert len(model.countries["FRA"].households.ts.historic("income")) == 1
    control = load_artifact(output / "control")
    assert control["quarters"][0]["completed_row"] == 2
    assert len(control["passes"]) == 2


def test_native_numeric_series_keeps_industry_labels_and_name(output):
    import pandas as pd

    series = pd.Series([2.0, np.nan], index=pd.Index(["A", "B"], name="Industry"), name="Household Consumption in LCU")
    save_artifact(output, {"industry_consumption": series})
    restored = load_artifact(output)["industry_consumption"]
    pd.testing.assert_series_equal(restored, series)
