from types import SimpleNamespace

import numpy as np
import pytest

from macromodel.agents.individuals.individual_properties import ActivityStatus
from macromodel.country.country import Country
from macromodel.timeseries import TimeSeries
from macromodel.util.wage_receipts import labour_income_components, wage_anchor_factor, wage_price_factor


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


# --- inflation-target anchor of wage indexation ---------------------------------------------------------


def test_unit_anchor_factor_leaves_wage_price_factor_bit_identical():
    assert wage_price_factor(1.5, 0.9) == 1.5**0.9
    assert wage_price_factor(1.5, 0.9, 1.0) == 1.5**0.9


@pytest.mark.parametrize("elapsed_periods", [0, 1, 40])
def test_anchor_has_no_effect_at_full_indexation(elapsed_periods):
    assert wage_anchor_factor(0.02, 1.0, elapsed_periods, 4) == 1.0


def test_anchor_is_neutral_at_the_first_period():
    assert wage_anchor_factor(0.02, 0.9, 0, 4) == 1.0


def test_anchor_reaches_the_annual_target_over_one_year_when_cpi_has_no_weight():
    assert wage_anchor_factor(0.02, 0.0, 4, 4) == pytest.approx(1.02)


@pytest.mark.parametrize("psi", [0.0, 0.5, 0.9])
@pytest.mark.parametrize("periods_per_year", [4, 12])
def test_anchor_growth_per_period_is_one_minus_psi_times_the_target(psi, periods_per_year):
    growth = wage_anchor_factor(0.02, psi, 8, periods_per_year) / wage_anchor_factor(0.02, psi, 7, periods_per_year)
    assert growth == pytest.approx(1.02 ** ((1 - psi) / periods_per_year))


def test_anchored_wage_factor_growth_mixes_lagged_cpi_and_target():
    psi, target, ppy = 0.9, 0.02, 4

    def factor(n, cpi):
        return wage_price_factor(cpi, psi, wage_anchor_factor(target, psi, n, ppy))

    # CPI flat: only the target moves the factor, with weight 1 - psi.
    assert np.log(factor(5, 1.0) / factor(4, 1.0)) == pytest.approx((1 - psi) * np.log(1 + target) / ppy)
    # CPI +1% in a period: lagged inflation enters with weight psi on top of the target term.
    assert np.log(factor(5, 1.01) / factor(4, 1.0)) == pytest.approx(
        psi * np.log(1.01) + (1 - psi) * np.log(1 + target) / ppy
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(target_annual_inflation=-1.0, elasticity=0.9, elapsed_periods=1, periods_per_year=4),
        dict(target_annual_inflation=float("nan"), elasticity=0.9, elapsed_periods=1, periods_per_year=4),
        dict(target_annual_inflation=0.02, elasticity=float("nan"), elapsed_periods=1, periods_per_year=4),
        dict(target_annual_inflation=0.02, elasticity=0.9, elapsed_periods=-1, periods_per_year=4),
        dict(target_annual_inflation=0.02, elasticity=0.9, elapsed_periods=1, periods_per_year=0),
    ],
)
def test_invalid_anchor_inputs_rejected(kwargs):
    with pytest.raises(ValueError):
        wage_anchor_factor(**kwargs)


@pytest.mark.parametrize("anchor", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_anchor_factor_rejected_in_wage_price_factor(anchor):
    with pytest.raises(ValueError):
        wage_price_factor(1.5, 0.9, anchor)


def make_anchor_country(anchor, psi=0.9, target=0.02, time_unit=3, cpi=1.2):
    """Country stub whose economy history length (completed periods + 1) can change after the lock."""
    history = [0.0] * 5  # n = 4 completed periods
    country = Country.__new__(Country)
    country.firms = SimpleNamespace(
        configuration=SimpleNamespace(
            parameters=SimpleNamespace(wage_cpi_indexation_elasticity=psi, wage_inflation_anchor=anchor)
        )
    )
    country.central_bank = SimpleNamespace(states={"targeted_inflation_rate": target})
    country.economy = SimpleNamespace(
        time_unit=time_unit,
        current_consumer_price_level=lambda: cpi,
        consumer_price_level_series_name=lambda: "cpi",
        ts=SimpleNamespace(historic=lambda name: history),
    )
    country.individuals = SimpleNamespace(
        states={"Activity Status": np.array([ActivityStatus.EMPLOYED, ActivityStatus.UNEMPLOYED])},
        ts=TimeSeries(employee_income=np.array([72.0, 1000.0])),
    )
    return country, history


def test_anchor_off_locks_a_unit_factor_and_reproduces_the_unanchored_receipts():
    country, _ = make_anchor_country(anchor=False)
    country._lock_wage_price_basis()
    assert country._wage_anchor_factor == 1.0
    assert country.nominal_employee_income()[0] == 72.0 * 1.2**0.9


def test_anchor_on_locks_target_factor_from_central_bank_target_and_period_count():
    country, _ = make_anchor_country(anchor=True)
    country._lock_wage_price_basis()
    # quarterly model (time_unit = 3, 4 periods a year), n = 4 completed periods, psi = 0.9
    expected = 1.02 ** (0.1 * 4 / 4)
    assert country._wage_anchor_factor == pytest.approx(expected)
    assert country.nominal_employee_income()[0] == pytest.approx(72.0 * 1.2**0.9 * expected)


def test_locked_anchor_does_not_move_when_the_period_counter_advances_but_the_preview_does():
    country, history = make_anchor_country(anchor=True)
    country._lock_wage_price_basis()
    locked_receipt = country.nominal_employee_income()[0]
    history.append(0.0)  # goods clearing appends a CPI observation after the lock
    assert country.nominal_employee_income()[0] == locked_receipt
    assert country._wage_inflation_anchor_factor() == pytest.approx(1.02 ** (0.1 * 5 / 4))


def test_anchor_factor_is_one_for_a_country_without_the_switch():
    country, _ = make_anchor_country(anchor=True)
    del country.firms.configuration.parameters.wage_inflation_anchor
    assert country._wage_inflation_anchor_factor() == 1.0
