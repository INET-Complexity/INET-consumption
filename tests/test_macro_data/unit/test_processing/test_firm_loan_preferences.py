from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from macro_data.processing.synthetic_firms.firm_tools import (
    function_parameters_dependent_initialisation,
)
from macro_data.processing.synthetic_matching.matching_firms_with_banks import (
    match_firms_with_banks_optimal,
)


def _initialize_firms(production, *, zero_debt=True, total_debt=100.0):
    firm_data = pd.DataFrame(
        {
            "Production": production,
            "Industry": np.zeros(len(production), dtype=int),
            "Price": np.ones(len(production)),
        }
    )
    stocks = function_parameters_dependent_initialisation(
        firm_data=firm_data,
        intermediate_inputs_productivity_matrix=np.ones((1, 1)),
        capital_input_use_matrix=np.zeros((1, 1)),
        capital_inputs_productivity_matrix=np.ones((1, 1)),
        total_firm_deposits=0.0,
        total_firm_debt=total_debt,
        assume_zero_initial_debt=zero_debt,
        assume_zero_initial_deposits=True,
        capital_inputs_utilisation_rate=1.0,
        initial_inventory_to_input_fraction=0.1,
        intermediate_inputs_utilisation_rate=1.0,
    )
    return firm_data, stocks


def test_zero_initial_debt_keeps_notional_firm_debt_from_capital_stock():
    firm_data, _ = _initialize_firms([1.0, 3.0])

    assert firm_data["Debt"].to_numpy() == pytest.approx([0.0, 0.0])
    assert firm_data["Notional Debt"].to_numpy() == pytest.approx([25.0, 75.0])


def test_zero_initial_debt_with_zero_aggregate_capital_stock_is_finite():
    firm_data, _ = _initialize_firms([0.0, 0.0])

    assert firm_data["Debt"].to_numpy() == pytest.approx([0.0, 0.0])
    assert firm_data["Notional Debt"].to_numpy() == pytest.approx([0.0, 0.0])
    assert np.isfinite(firm_data[["Debt", "Notional Debt", "Equity"]].to_numpy()).all()


def test_positive_debt_without_capital_stock_is_rejected():
    with pytest.raises(ValueError, match="aggregate capital stock is zero"):
        _initialize_firms([0.0], zero_debt=False, total_debt=100.0)


def test_multi_bank_matching_uses_notional_debt_without_creating_actual_loans():
    firms = SimpleNamespace(
        number_of_firms=2,
        firm_data=pd.DataFrame(
            {
                "Deposits": [0.0, 0.0],
                "Debt": [0.0, 0.0],
                "Notional Debt": [70.0, 30.0],
            }
        ),
    )
    banks = SimpleNamespace(
        number_of_banks=2,
        bank_data=pd.DataFrame(
            {
                "Deposits from Firms": [0.0, 0.0],
                "Loans to Firms": [0.0, 0.0],
                "Notional Loans to Firms": [50.0, 50.0],
            }
        ),
    )

    match_firms_with_banks_optimal(firms=firms, banks=banks)

    assert firms.firm_data["Debt"].to_numpy() == pytest.approx([0.0, 0.0])
    assert banks.bank_data["Loans to Firms"].to_numpy() == pytest.approx([0.0, 0.0])
    assert banks.bank_data["Notional Loans to Firms"].to_numpy() == pytest.approx([50.0, 50.0])
    assert sorted(firms.firm_data["Corresponding Bank ID"].to_numpy().tolist()) == [0, 1]
    assert sorted(len(ids) for ids in banks.bank_data["Corresponding Firms ID"]) == [1, 1]
