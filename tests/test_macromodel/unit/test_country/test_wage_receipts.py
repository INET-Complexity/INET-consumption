from types import SimpleNamespace

import numpy as np
import pytest

from macromodel.agents.individuals.individual_properties import ActivityStatus
from macromodel.country.country import Country
from macromodel.timeseries import TimeSeries
from macromodel.util.wage_receipts import labour_income_components, wage_price_factor


def make_country(cpi=1.5, elasticity=1.0):
    country = Country.__new__(Country)
    country._wage_price_level = cpi
    country.economy = SimpleNamespace(current_consumer_price_level=lambda: 9.0)
    country.firms = SimpleNamespace(
        configuration=SimpleNamespace(parameters=SimpleNamespace(wage_cpi_indexation_elasticity=elasticity))
    )
    country.individuals = SimpleNamespace(
        states={"Activity Status": np.array([ActivityStatus.EMPLOYED, ActivityStatus.UNEMPLOYED])},
        ts=TimeSeries(employee_income=np.array([72.0, 1000.0])),
    )
    return country


@pytest.mark.parametrize("elasticity", [0.0, 0.5, 1.0])
def test_locked_nominal_receipt_reconciles_payroll_and_employee_deductions(elasticity):
    country = make_country(elasticity=elasticity)
    receipts = country.nominal_employee_income()
    assert receipts[0] == pytest.approx(72 * 1.5**elasticity)
    assert receipts[1] == 0
    bridge = labour_income_components(receipts, 0.2, 0.1, 0.3)
    np.testing.assert_allclose(bridge["gross_wages"], receipts + bridge["income_tax"] + bridge["employee_si"])
    np.testing.assert_allclose(
        bridge["employer_cost"], receipts + bridge["income_tax"] + bridge["employee_si"] + bridge["employer_si"]
    )
    # A change to closing CPI cannot retroactively change agreed cash wages.
    assert country.nominal_employee_income()[0] == receipts[0]


def test_nominal_wage_then_one_deflation_recovers_real_learning_signal():
    country = make_country()
    nominal = country.nominal_employee_income()[0]
    benefits_real = 20
    assert (nominal + 1.5 * benefits_real) / 1.5 == pytest.approx(92)


@pytest.mark.parametrize("cpi", [0, -1, float("nan"), float("inf")])
def test_invalid_wage_price_level_rejected(cpi):
    with pytest.raises(ValueError):
        wage_price_factor(cpi)
