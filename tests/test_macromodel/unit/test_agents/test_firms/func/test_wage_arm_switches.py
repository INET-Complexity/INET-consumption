"""Characterization tests for the incumbent wage-rule arm switches.

Pins that every switch defaults to the historical formula
``(1 + m) * TFP * u * w_init / tax`` and that each flag removes exactly one
factor. See wiki/experiments/2026-08-28-wage-arm-multiseed-inflation-attribution.
"""

import numpy as np
import pytest

from macromodel.agents.firms.func.wage_setter import WorkEffortFirmWageSetter

N_FIRMS = 3
INCOME_TAX = 0.2
EMPLOYEE_SI = 0.1
EMPLOYER_SI = 0.3

MARKUP = np.array([0.02, 0.05, 0.00])
TFP = np.array([1.05, 1.10, 1.00])
EFFORT = np.array([1.10, 1.20, 0.90])
INITIAL_WAGE = 50.0
# Firm of each employed individual; the fifth individual is unemployed.
FIRM_OF_EMPLOYEE = np.array([0, 0, 1, 2])

TAX_GROSS_UP = (1.0 + EMPLOYER_SI) / (1 - EMPLOYEE_SI - INCOME_TAX * (1 - EMPLOYEE_SI))


def _setter(**kwargs) -> WorkEffortFirmWageSetter:
    return WorkEffortFirmWageSetter(
        labour_market_tightness_markup_scale=0.05,
        markup_time_span=4,
        **kwargs,
    )


def _incomes(setter: WorkEffortFirmWageSetter) -> np.ndarray:
    return setter.set_employee_income(
        corresponding_firm=np.array([0, 0, 1, 2, -1]),
        current_individual_labour_inputs=np.array([1.0, 1.0, 1.0, 1.0, 0.0]),
        current_individual_stating_new_job=np.zeros(5, dtype=bool),
        current_employee_income=np.full(5, 100.0),
        current_individual_offered_wage=np.full(5, 111.0),
        current_target_production=np.full(N_FIRMS, 10.0),
        current_limiting_intermediate_inputs=np.full(N_FIRMS, 10.0),
        current_limiting_capital_inputs=np.full(N_FIRMS, 10.0),
        labour_inputs_from_employees=np.full(N_FIRMS, 1.0),
        industry_labour_productivity_by_firm=np.full(N_FIRMS, 1.0),
        initial_wage_per_capita=np.full(N_FIRMS, INITIAL_WAGE),
        current_wage_per_capita=np.full(N_FIRMS, 55.0),
        current_labour_productivity_factor=EFFORT,
        prev_labour_productivity_factor=np.ones(N_FIRMS),
        current_wage_tightness_markup=MARKUP,
        estimated_ppi_inflation=0.01,
        income_taxes=INCOME_TAX,
        employee_social_insurance_tax=EMPLOYEE_SI,
        employer_social_insurance_tax=EMPLOYER_SI,
        current_tfp_multiplier=TFP,
    )


@pytest.mark.parametrize(
    ("flags", "expected_by_firm"),
    [
        ({}, (1 + MARKUP) * TFP * EFFORT * INITIAL_WAGE),
        ({"incumbent_tightness_markup": False}, TFP * EFFORT * INITIAL_WAGE),
        ({"incumbent_effort_indexation": False}, (1 + MARKUP) * TFP * INITIAL_WAGE),
        ({"incumbent_tfp_indexation": False}, (1 + MARKUP) * EFFORT * INITIAL_WAGE),
        (
            {"incumbent_effort_indexation": False, "incumbent_tfp_indexation": False},
            (1 + MARKUP) * INITIAL_WAGE,
        ),
    ],
    ids=["default", "no_markup", "no_effort", "no_tfp", "no_effort_no_tfp"],
)
def test_incumbent_wage_arm_removes_exactly_the_named_factor(flags, expected_by_firm):
    """Defaults reproduce the historical formula; each flag drops one factor."""
    got = _incomes(_setter(**flags))[:4]
    expected = (expected_by_firm / TAX_GROSS_UP)[FIRM_OF_EMPLOYEE]
    np.testing.assert_allclose(got, expected)


def test_unemployed_individual_receives_no_income():
    assert _incomes(_setter())[4] == 0.0
