"""Unit booking checks using native income and bank-account methods, no model run."""

import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

RUN_MODEL = Path(__file__).resolve().parents[3] / "run_model"
if str(RUN_MODEL) not in sys.path:
    sys.path.insert(0, str(RUN_MODEL))

from src.experiments.income_transfer import OneTimeIncomeTransfer  # noqa: E402
from src.experiments.paired_checkpoint import PairedCheckpoint  # noqa: E402

from macromodel.agents.banks.banks import Banks  # noqa: E402
from macromodel.country.country import Country  # noqa: E402
from macromodel.timeseries import TimeSeries  # noqa: E402


@pytest.fixture
def model():
    hh = SimpleNamespace(
        ts=TimeSeries(
            income=np.array([10.0, 20.0, 30.0]),
            expected_income=np.array([10.0, 20.0, 30.0]),
            liquid_financial_assets=np.array([100.0, -20.0, 300.0]),
        ),
        states={"corr_individuals": [[0], [1], [2]], "Corresponding Bank ID": np.array([0, 1, 0])},
        compute_income=lambda: np.array([10.0, 20.0, 30.0]),
    )
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
        )
    )
    banks.compute_deposits = MethodType(Banks.compute_deposits, banks)
    country = SimpleNamespace(households=hh, banks=banks, _staged_household_income_shock=None)
    country.stage_household_income_shock = MethodType(Country.stage_household_income_shock, country)
    country._apply_household_income_shock = MethodType(Country._apply_household_income_shock, country)
    country._clear_household_income_shock = MethodType(Country._clear_household_income_shock, country)
    return SimpleNamespace(countries={"FRA": country}, prehooks=[], posthooks=[])


def install(model, amounts=(2.0, 4.0, 6.0)):
    transfer = OneTimeIncomeTransfer("FRA", year=2019, month=1, amounts_nominal=amounts)
    transfer.install(model)
    return transfer


def settle_native_seams(model):
    country = model.countries["FRA"]
    hh, banks = country.households, country.banks
    # Native receipt is applied once to freshly computed expected/realised bases.
    expected = country._apply_household_income_shock(hh.compute_income())
    for _ in range(2):  # Both pass reconstructions, not cumulative additions.
        hh.ts.override_current("expected_income", country._apply_household_income_shock(hh.compute_income()))
        np.testing.assert_array_equal(hh.ts.current("expected_income"), expected)
    income = country._apply_household_income_shock(hh.compute_income())
    receipt = country._staged_household_income_shock.copy()
    hh.ts.income.append(income)
    country._clear_household_income_shock()
    # Spend half the receipt: gross fund outflow must NOT be inferred from ΔLFA.
    hh.ts.liquid_financial_assets.append(hh.ts.current("liquid_financial_assets") + receipt / 2)
    Banks.update_deposits(
        banks,
        current_firm_deposits=np.array([30.0, 40.0]),
        current_household_deposits=hh.ts.current("liquid_financial_assets"),
        firm_corresponding_bank=np.array([0, 1]),
        households_corresponding_bank=hh.states["Corresponding Bank ID"],
    )
    banks.ts.deposits.append(banks.compute_deposits())


def test_one_time_receipt_and_gross_payer_booking(model):
    original = model.countries["FRA"].households.ts.current("liquid_financial_assets").copy()
    transfer = install(model)
    transfer.stage(model, 2018, 10)
    assert model.countries["FRA"]._staged_household_income_shock is None
    transfer.stage(model, 2019, 1)
    transfer.stage(model, 2019, 1)
    np.testing.assert_array_equal(model.countries["FRA"].households.ts.current("liquid_financial_assets"), original)
    assert transfer.booking is None
    settle_native_seams(model)
    transfer.verify_settlement(model, 20, 2019, 1)
    booking = transfer.booking
    assert booking.payer_outflow_nominal == 12.0
    np.testing.assert_array_equal(booking.household_receipts_nominal, [2, 4, 6])
    np.testing.assert_array_equal(booking.bank_funding_nominal, [8, 4])
    assert booking.bank_funding_nominal.sum() == booking.payer_outflow_nominal
    assert not booking.household_receipts_nominal.flags.writeable
    transfer.verify_settlement(model, 20, 2019, 1)
    transfer.stage(model, 2019, 4)
    assert transfer.booking is booking
    assert model.countries["FRA"]._staged_household_income_shock is None


def test_zero_control_is_valid(model):
    transfer = install(model, [0, 0, 0])
    transfer.stage(model, 2019, 1)
    settle_native_seams(model)
    transfer.verify_settlement(model, 20, 2019, 1)
    assert transfer.booking.payer_outflow_nominal == 0


@pytest.mark.parametrize("values", [[-1, 0], [np.nan], [np.inf], [], [[1]], [1e308, 1e308]])
def test_invalid_amounts(values):
    with pytest.raises(ValueError):
        OneTimeIncomeTransfer("FRA", year=2019, month=1, amounts_nominal=values)


@pytest.mark.parametrize("year,month", [(True, 1), (2019, 0), (2019, 13), (2019, 1.5)])
def test_invalid_dates(year, month):
    with pytest.raises(ValueError):
        OneTimeIncomeTransfer("FRA", year=year, month=month, amounts_nominal=[1])


def test_amounts_copied_and_wrong_dimensions_rejected(model):
    amounts = np.ones(3)
    transfer = install(model, amounts)
    amounts[:] = 100
    np.testing.assert_array_equal(transfer.amounts_nominal, [1, 1, 1])
    other = OneTimeIncomeTransfer("FRA", year=2019, month=1, amounts_nominal=[1])
    with pytest.raises(ValueError, match="dimensions"):
        other.install(model)


def test_reinstall_other_arm_and_reorder_rejected(model):
    other = PairedCheckpoint(model).clone()
    transfer = install(model)
    with pytest.raises(ValueError, match="already installed"):
        transfer.install(other)
    with pytest.raises(ValueError, match="original arm"):
        transfer.stage(other, 2019, 1)
    model.countries["FRA"].households.states["corr_individuals"].reverse()
    with pytest.raises(ValueError, match="ordering"):
        transfer.stage(model, 2019, 1)


def test_missed_date_and_duplicate_native_stage_fail_without_debit(model):
    transfer = install(model)
    with pytest.raises(RuntimeError, match="missed"):
        transfer.stage(model, 2019, 4)
    country = model.countries["FRA"]
    country.stage_household_income_shock(np.ones(3))
    with pytest.raises(ValueError, match="already staged"):
        transfer.stage(model, 2019, 1)
    assert transfer.booking is None
    assert transfer._period is None


@pytest.mark.parametrize("fault", ["pending", "income", "bank", "balance", "extra_row", "nonfinite_balance"])
def test_unreconciled_settlement_never_records_payer_debit(model, fault):
    transfer = install(model)
    transfer.stage(model, 2019, 1)
    if fault != "pending":
        settle_native_seams(model)
    country = model.countries["FRA"]
    if fault == "income":
        country.households.ts.current("income")[:] += 2
    elif fault == "bank":
        country.banks.ts.current("deposits_from_households")[:] += 1
    elif fault == "balance":
        country.banks.ts.current("deposits")[:] += 1
    elif fault == "nonfinite_balance":
        country.banks.ts.current("equity")[:] = np.inf
        country.banks.ts.current("deposits")[:] = np.inf
    elif fault == "extra_row":
        country.households.ts.income.append(np.zeros(3))
    with pytest.raises(RuntimeError):
        transfer.verify_settlement(model, 20, 2019, 1)
    assert transfer.booking is None


def test_native_simulation_settles_receipt_in_deterministic_fixture(datawrapper):
    from macromodel.configurations import CountryConfiguration, SimulationConfiguration
    from macromodel.simulation import Simulation

    simulation = Simulation.from_datawrapper(
        datawrapper=datawrapper,
        simulation_configuration=SimulationConfiguration(
            country_configurations={"FRA": CountryConfiguration()},
            seed=15,
        ),
    )
    amounts = np.full(simulation.countries["FRA"].households.ts.current("n_households"), 2.5)
    transfer = OneTimeIncomeTransfer(
        "FRA",
        year=simulation.timestep.year,
        month=simulation.timestep.month,
        amounts_nominal=amounts,
    )
    transfer.install(simulation)
    simulation.iterate(t=0)
    assert transfer.booking is not None
    assert transfer.booking.payer_outflow_nominal == float(amounts.sum())
    assert simulation.countries["FRA"]._staged_household_income_shock is None
