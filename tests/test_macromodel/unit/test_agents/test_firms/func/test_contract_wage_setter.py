"""Unit tests for ContractWageSetter.

Pins the settled specification in
knowledge-vault/raw/plans/2026-08-28-wage-rule-restructure-and-inflation-gain-decomposition.md
section 4c:

    stored:  w_rate[j,t] = w_rate[j,t-1] * TFP[i,t] / TFP[i,t-1]
    paid:    earnings[j,t] = w_rate[j,t] * u[i,t]

No (1+m) and no u in the stored rate; new hires adopt the accepted offer rate
and keep it.
"""

import numpy as np
import pytest

from macromodel.agents.firms.func.wage_setter import ContractWageSetter

N_FIRMS = 3
INCOME_TAX = 0.2
EMPLOYEE_SI = 0.1
EMPLOYER_SI = 0.3

# Individual 4 is unemployed.
CORRESPONDING_FIRM = np.array([0, 0, 1, 2, -1])
FIRM_OF_EMPLOYEE = np.array([0, 0, 1, 2])

MARKUP = np.array([0.02, 0.05, 0.00])
EFFORT = np.array([1.10, 1.20, 0.90])
PREV_EFFORT = np.array([1.00, 1.00, 1.00])
TFP = np.array([1.05, 1.10, 1.00])
PREV_TFP = np.array([1.00, 1.00, 1.00])
INITIAL_WAGE = np.full(N_FIRMS, 50.0)

TAX_GROSS_UP = (1.0 + EMPLOYER_SI) / (1 - EMPLOYEE_SI - INCOME_TAX * (1 - EMPLOYEE_SI))


def _setter(**kwargs) -> ContractWageSetter:
    return ContractWageSetter(
        labour_market_tightness_markup_scale=0.05,
        markup_time_span=4,
        **kwargs,
    )


def _run(setter, carried, new_job=None, offered=None, tfp=TFP, prev_tfp=PREV_TFP):
    if new_job is None:
        new_job = np.zeros(5, dtype=bool)
    if offered is None:
        offered = np.full(5, 111.0)
    return setter.set_employee_income(
        corresponding_firm=CORRESPONDING_FIRM,
        current_individual_labour_inputs=np.array([1.0, 1.0, 1.0, 1.0, 0.0]),
        current_individual_stating_new_job=new_job,
        current_employee_income=np.full(5, 100.0),
        current_individual_offered_wage=offered,
        current_target_production=np.full(N_FIRMS, 10.0),
        current_limiting_intermediate_inputs=np.full(N_FIRMS, 10.0),
        current_limiting_capital_inputs=np.full(N_FIRMS, 10.0),
        labour_inputs_from_employees=np.full(N_FIRMS, 1.0),
        industry_labour_productivity_by_firm=np.full(N_FIRMS, 1.0),
        initial_wage_per_capita=INITIAL_WAGE,
        current_wage_per_capita=np.full(N_FIRMS, 55.0),
        current_labour_productivity_factor=EFFORT,
        prev_labour_productivity_factor=PREV_EFFORT,
        current_wage_tightness_markup=MARKUP,
        estimated_ppi_inflation=0.01,
        income_taxes=INCOME_TAX,
        employee_social_insurance_tax=EMPLOYEE_SI,
        employer_social_insurance_tax=EMPLOYER_SI,
        current_tfp_multiplier=tfp,
        prev_tfp_multiplier=prev_tfp,
        carried_wage_rate=carried,
    )


def test__carried_rate_is_required():
    with pytest.raises(ValueError, match="carried_wage_rate"):
        _run(_setter(), carried=None)


def test__invalid_initial_rate_source_is_rejected():
    with pytest.raises(ValueError, match="initial_rate_source"):
        _setter(initial_rate_source="nonsense")


def test__incumbent_rate_indexes_to_tfp_only():
    """No (1+m) and no u in the stored rate."""
    carried = np.array([10.0, 20.0, 30.0, 40.0, 5.0])
    _run(_setter(), carried)

    expected = np.array([10.0, 20.0, 30.0, 40.0]) * (TFP / PREV_TFP)[FIRM_OF_EMPLOYEE]
    np.testing.assert_allclose(carried[:4], expected)


def test__paid_earnings_are_rate_times_effort():
    carried = np.array([10.0, 20.0, 30.0, 40.0, 5.0])
    earnings = _run(_setter(), carried)

    np.testing.assert_allclose(earnings[:4], carried[:4] * EFFORT[FIRM_OF_EMPLOYEE])


def test__wage_bill_is_invariant_to_where_effort_is_applied():
    """sum_j(w*u) == u*sum_j(w) for a firm-level u, so the bill is unchanged."""
    carried = np.array([10.0, 20.0, 30.0, 40.0, 5.0])
    earnings = _run(_setter(), carried)

    for firm_id in range(N_FIRMS):
        members = FIRM_OF_EMPLOYEE == firm_id
        rate_then_effort = earnings[:4][members].sum()
        effort_then_rate = EFFORT[firm_id] * carried[:4][members].sum()
        np.testing.assert_allclose(rate_then_effort, effort_then_rate)


def test__unemployed_receive_no_earnings_and_keep_their_rate():
    carried = np.array([10.0, 20.0, 30.0, 40.0, 5.0])
    earnings = _run(_setter(), carried)

    assert earnings[4] == 0.0
    assert carried[4] == 5.0


def test__new_hire_adopts_the_offered_rate_and_keeps_it():
    """The offer must persist, not evaporate after one period."""
    setter = _setter()
    carried = np.array([10.0, 20.0, 30.0, 40.0, 5.0])
    new_job = np.array([False, True, False, False, False])
    offered = np.array([0.0, 77.0, 0.0, 0.0, 0.0])

    _run(setter, carried, new_job=new_job, offered=offered)
    assert carried[1] == 77.0

    # Next period the worker is no longer new; the rate persists and indexes.
    _run(setter, carried, tfp=np.array([2.10, 2.20, 2.00]), prev_tfp=TFP)
    np.testing.assert_allclose(carried[1], 77.0 * (2.20 / 1.10))


def test__firm_anchor_arm_overwrites_dispersion_once():
    carried = np.array([10.0, 20.0, 30.0, 40.0, 5.0])
    _run(_setter(initial_rate_source="firm_anchor"), carried)

    anchored = INITIAL_WAGE[FIRM_OF_EMPLOYEE] / TAX_GROSS_UP
    np.testing.assert_allclose(carried[:4], anchored * (TFP / PREV_TFP)[FIRM_OF_EMPLOYEE])


def test__individual_arm_preserves_initial_dispersion():
    """Two workers at the same firm keep distinct rates."""
    setter = _setter(initial_rate_source="individual")
    carried = np.array([10.0, 20.0, 30.0, 40.0, 5.0])
    _run(setter, carried)

    # Individuals 0 and 1 are both at firm 0 and started 10 vs 20.
    assert carried[0] != carried[1]
    np.testing.assert_allclose(carried[1] / carried[0], 2.0)


def test__seeding_happens_only_once():
    setter = _setter(initial_rate_source="firm_anchor")
    carried = np.array([10.0, 20.0, 30.0, 40.0, 5.0])
    _run(setter, carried)
    after_first = carried.copy()

    _run(setter, carried, tfp=TFP, prev_tfp=TFP)
    np.testing.assert_allclose(carried, after_first)


def test__zero_previous_tfp_does_not_produce_nan():
    carried = np.array([10.0, 20.0, 30.0, 40.0, 5.0])
    _run(_setter(), carried, prev_tfp=np.zeros(N_FIRMS))

    assert np.all(np.isfinite(carried))
    np.testing.assert_allclose(carried[:4], np.array([10.0, 20.0, 30.0, 40.0]))
