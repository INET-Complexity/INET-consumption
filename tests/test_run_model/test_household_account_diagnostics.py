import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest

RUN_MODEL_PATH = Path(__file__).resolve().parents[2] / "run_model"
if str(RUN_MODEL_PATH) not in sys.path:
    sys.path.insert(0, str(RUN_MODEL_PATH))

from src.household_account_diagnostics import (  # noqa: E402
    build_household_account_panel,
    compute_household_accounts_from_rules,
    make_paper_account_panel,
    run_household_account_diagnostics,
    write_household_account_outputs,
)

from macro_data import DataWrapper  # noqa: E402


def _household_data():
    return pd.DataFrame(
        {
            "Wealth in Deposits": [10.0, 20.0, 30.0],
            "Mutual Funds": [1.0, 2.0, 3.0],
            "Bonds": [1.0, 2.0, 3.0],
            "Shares": [1.0, 2.0, 3.0],
            "Managed Accounts": [1.0, 2.0, 3.0],
            "Money owed to Households": [1.0, 2.0, 3.0],
            "Other Assets": [1.0, 2.0, 3.0],
            "Voluntary Pension": [1.0, 2.0, 3.0],
            "Value of the Main Residence": [100.0, 200.0, 300.0],
            "Value of Other Non-Business Real Estate": [10.0, 20.0, 30.0],
            "Outstanding Balance of Mortgage Debt": [40.0, 50.0, 60.0],
            "Outstanding Balance of Non-Mortgage Debt": [5.0, 6.0, 7.0],
        }
    )


def _datawrapper(household_data=None):
    population = SimpleNamespace(household_data=household_data if household_data is not None else _household_data())
    country = SimpleNamespace(population=population)
    return SimpleNamespace(synthetic_countries={"FRA": country})


def _write_datawrapper_pickle(path, datawrapper):
    state = {
        "synthetic_countries": datawrapper.synthetic_countries,
        "synthetic_rest_of_the_world": None,
        "exchange_rates": pd.DataFrame(),
        "origin_trade_proportions": pd.DataFrame(),
        "destination_trade_proportions": pd.DataFrame(),
        "configuration": SimpleNamespace(),
        "calibration_data": pd.DataFrame(),
        "industries": [],
        "emission_factors": {},
        "emissions_energy_factors": None,
        "aggregation_structure": None,
        "time_unit": 3,
    }
    with open(path, "wb") as handle:
        pickle.dump(state, handle)


def _write_h5(path, *, n_households=3, cpi_fixed=None, cpi_transaction=None, income=None):
    if cpi_fixed is None:
        cpi_fixed = np.array([1.0, 2.0])
    if cpi_transaction is None:
        cpi_transaction = np.array([1.0, 4.0])
    if income is None:
        income = np.array([[100.0, 0.0, -10.0], [110.0, 0.0, -20.0]])[:, :n_households]
    shape = income.shape
    values = {
        "income": income,
        "consumption": np.array([[50.0, 60.0, 70.0], [55.0, 66.0, 77.0]])[:, :n_households],
        "wealth_deposits": np.array([[9.0, 19.0, 29.0], [10.0, 20.0, 30.0]])[:, :n_households],
        "wealth_other_financial_assets": np.array([[6.0, 12.0, 18.0], [7.0, 14.0, 21.0]])[:, :n_households],
        "wealth_main_residence": np.array([[90.0, 190.0, 290.0], [100.0, 200.0, 300.0]])[:, :n_households],
        "wealth_other_properties": np.array([[9.0, 19.0, 29.0], [10.0, 20.0, 30.0]])[:, :n_households],
        "wealth_other_real_assets": np.array([[4.0, 5.0, 6.0], [8.0, 9.0, 10.0]])[:, :n_households],
        "mortgage_debt": np.array([[35.0, 45.0, 55.0], [40.0, 50.0, 60.0]])[:, :n_households],
        "consumption_loan_debt": np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])[:, :n_households],
        "debt": np.array([[36.0, 47.0, 58.0], [42.0, 53.0, 64.0]])[:, :n_households],
        "net_wealth": np.array([[82.0, 198.0, 314.0], [93.0, 210.0, 327.0]])[:, :n_households],
    }
    assert all(value.shape == shape for value in values.values())
    with h5py.File(path, "w") as handle:
        country = handle.create_group("FRA")
        economy = country.create_group("economy")
        economy.create_dataset("cpi_fixed_basket", data=np.asarray(cpi_fixed).reshape(-1, 1))
        economy.create_dataset("cpi_transaction", data=np.asarray(cpi_transaction).reshape(-1, 1))
        households = country.create_group("households")
        for field, value in values.items():
            households.create_dataset(field, data=value)


def test_build_household_account_panel_joins_paper_and_runtime_accounts(tmp_path):
    h5_path = tmp_path / "model.h5"
    _write_h5(h5_path)

    panel, metadata = build_household_account_panel(_datawrapper(), h5_path, "FRA")

    assert metadata["period"] == 1
    assert panel["household_id"].tolist() == [0, 1, 2]
    assert panel.loc[0, "lfa"] == 10.0
    assert panel.loc[0, "ifa"] == 7.0
    assert panel.loc[0, "ha"] == 110.0
    assert panel.loc[0, "mr"] == 40.0
    assert panel.loc[0, "db"] == 5.0
    assert panel.loc[0, "nw"] == 82.0
    assert panel.loc[0, "model_spendable_income"] == 110.0
    assert panel.loc[0, "real_lfa_fixed_cpi"] == 5.0
    assert panel.loc[0, "real_lfa_transaction_cpi"] == 2.5
    assert panel.loc[0, "runtime_other_real_assets"] == 8.0
    assert panel.loc[0, "runtime_net_wealth_identity"] == 93.0
    assert panel.loc[0, "consumption_to_income"] == pytest.approx(55.0 / 110.0)
    assert np.isnan(panel.loc[1, "consumption_to_income"])
    assert np.isnan(panel.loc[2, "paper_debt_to_income"])


def test_make_paper_account_panel_preserves_helper_index_as_household_id():
    household_data = _household_data()
    household_data.index = pd.Index([10, 11, 12], name="New Household ID")

    panel = make_paper_account_panel(_datawrapper(household_data), "FRA")

    assert panel["household_id"].tolist() == [10, 11, 12]


def test_stage_0_yaml_rules_drive_account_construction():
    stage_0_parameters = {
        "account_rules": {
            "lfa": {"components": [["Wealth in Deposits"]]},
            "ifa": {"components": [["Mutual Funds"]]},
            "ha": {"components": [["Value of the Main Residence"]]},
            "mr": {"preferred": ["Outstanding Balance of Mortgage Debt"]},
            "db": {"preferred": ["Outstanding Balance of Non-Mortgage Debt"]},
        }
    }

    accounts = compute_household_accounts_from_rules(_household_data(), stage_0_parameters)

    assert accounts.loc[0, "ha"] == 100.0
    assert accounts.loc[0, "nw"] == 66.0


def test_diagnostics_build_moments_quantiles_reconciliation_and_outputs(tmp_path):
    h5_path = tmp_path / "model.h5"
    output_dir = tmp_path / "diagnostics"
    _write_h5(h5_path)

    diagnostics = run_household_account_diagnostics(
        datawrapper=_datawrapper(),
        model_h5=h5_path,
        country_code="FRA",
    )
    paths = write_household_account_outputs(diagnostics, output_dir, country_code="FRA")

    assert set(paths) == {"panel", "moments", "quantiles", "reconciliation", "summary"}
    assert all(path.exists() for path in paths.values())
    assert "lfa" in diagnostics.moments["variable"].values
    assert "q0.5" in diagnostics.quantiles.columns
    reconciliation = diagnostics.reconciliation.set_index("check")["value"]
    assert float(reconciliation["max_abs_nw_identity_error"]) == pytest.approx(0.0)
    assert float(reconciliation["max_abs_runtime_net_wealth_identity_gap"]) == pytest.approx(0.0)
    assert "consumption_loan_debt" in paths["summary"].read_text()


def test_runner_loads_datawrapper_pickle_and_writes_outputs(tmp_path):
    data_pkl = tmp_path / "data.pkl"
    h5_path = tmp_path / "model.h5"
    _write_datawrapper_pickle(data_pkl, _datawrapper())
    _write_h5(h5_path)

    from run_household_account_diagnostics import main  # noqa: E402

    paths = main(data_pkl=data_pkl, model_h5=h5_path, country_code="FRA", output_dir=tmp_path / "out")

    assert paths["panel"].exists()
    loaded = DataWrapper.init_from_pickle(data_pkl)
    assert "FRA" in loaded.synthetic_countries


def test_dimension_mismatch_raises_clear_error(tmp_path):
    h5_path = tmp_path / "model.h5"
    _write_h5(h5_path, n_households=2)

    with pytest.raises(ValueError, match="Household-count mismatch"):
        build_household_account_panel(_datawrapper(), h5_path, "FRA")


def test_non_positive_or_non_finite_cpi_raises_clear_error(tmp_path):
    h5_path = tmp_path / "model.h5"
    _write_h5(h5_path, cpi_fixed=np.array([1.0, 0.0]))

    with pytest.raises(ValueError, match="cpi_fixed_basket must be finite and strictly positive"):
        build_household_account_panel(_datawrapper(), h5_path, "FRA")


def test_missing_source_account_columns_fail_loudly(tmp_path):
    h5_path = tmp_path / "model.h5"
    _write_h5(h5_path)

    with pytest.raises(ValueError, match="required HFCS source columns"):
        build_household_account_panel(_datawrapper(pd.DataFrame({"Income": [1.0, 2.0, 3.0]})), h5_path, "FRA")
