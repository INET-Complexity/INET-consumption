import numpy as np
import pandas as pd
import pytest

from macro_data.configuration.region import Region
from macro_data.processing.synthetic_firms.firm_tools import (
    function_parameters_dependent_initialisation,
    productivity_of_installed_capital,
)
from macro_data.readers.util.industry_extraction import _labour_compensation_lcu


def _initialise(capital_productivity: np.ndarray, utilisation: float) -> np.ndarray:
    firm_data = pd.DataFrame({"Industry": [0, 1], "Production": [10.0, 20.0], "Price": [1.0, 1.0]})
    stock, *_ = function_parameters_dependent_initialisation(
        firm_data,
        np.full((2, 2), 4.0),
        np.zeros((2, 2)),
        capital_productivity,
        total_firm_deposits=0.0,
        total_firm_debt=0.0,
        assume_zero_initial_debt=True,
        assume_zero_initial_deposits=True,
        capital_inputs_utilisation_rate=utilisation,
        initial_inventory_to_input_fraction=0.0,
        intermediate_inputs_utilisation_rate=1.0,
    )
    return stock


def test__initial_stock_equals_installed_capital_and_capacity_keeps_slack():
    data_productivity = np.array([[2.0, 4.0], [5.0, 8.0]])
    production = np.array([10.0, 20.0])
    utilisation = 0.85
    productivity = productivity_of_installed_capital(data_productivity, utilisation)

    stock = _initialise(productivity, utilisation)

    installed = (production / data_productivity[:, [0, 1]]).T
    assert stock == pytest.approx(installed)
    capacity = (productivity[:, [0, 1]].T * stock).min(axis=1)
    assert capacity == pytest.approx(production / utilisation)


def test__infinite_productivity_stays_infinite():
    assert productivity_of_installed_capital(np.array([[np.inf]]), 0.85)[0, 0] == np.inf


def _vectors() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Labour Output Ratio": [0.5, 0.05],
            "Output in LCU": [100.0, 80.0],
            "Labour Compensation": [200.0, 16.0],
        },
        index=["K", "L"],
    )


def test__real_estate_labour_keeps_d1_level_after_rents_leave_output():
    labour = _labour_compensation_lcu("FRA", _vectors(), yearly_factor=4.0)
    assert labour == pytest.approx([50.0, 16.0e6 / 4.0])


def test__regions_keep_the_labour_ratio():
    region = Region.from_code("FRA_FR1")
    labour = _labour_compensation_lcu(region, _vectors(), yearly_factor=4.0)
    assert labour == pytest.approx([50.0, 4.0])
