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


def _run(setter, carried, new_job=None, offered=None, tfp=TFP, prev_tfp=PREV_TFP, realised=None):
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
        realised_productivity_ratio=realised,
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


def _offer(
    setter,
    carried,
    markup=MARKUP,
    effort=EFFORT,
    prev_effort=PREV_EFFORT,
    tfp=TFP,
    prev_tfp=PREV_TFP,
    benefits=0.0,
    corresponding_firm=CORRESPONDING_FIRM,
):
    return setter.get_offered_wage_given_labour_inputs_function(
        corresponding_firm=corresponding_firm,
        current_individual_labour_inputs=np.array([1.0, 1.0, 1.0, 1.0, 0.0]),
        previous_employee_income=np.full(5, 999.0),  # must be ignored
        current_target_production=np.full(N_FIRMS, 10.0),
        current_limiting_intermediate_inputs=np.full(N_FIRMS, 10.0),
        current_limiting_capital_inputs=np.full(N_FIRMS, 10.0),
        industry_labour_productivity_by_firm=np.full(N_FIRMS, 1.0),
        initial_wage_per_capita=INITIAL_WAGE,
        current_wage_per_capita=np.full(N_FIRMS, 55.0),
        current_labour_productivity_factor=effort,
        prev_labour_productivity_factor=prev_effort,
        current_wage_tightness_markup=markup,
        income_taxes=INCOME_TAX,
        employee_social_insurance_tax=EMPLOYEE_SI,
        employer_social_insurance_tax=EMPLOYER_SI,
        unemployment_benefits_by_individual=benefits,
        current_tfp_multiplier=tfp,
        prev_tfp_multiplier=prev_tfp,
        carried_wage_rate=carried,
    )


def test__offer_requires_carried_rate():
    with pytest.raises(ValueError, match="carried_wage_rate"):
        _offer(_setter(), carried=None)


def test__offer_reference_is_firm_average_carried_rate():
    """w_offer = mean(carried rate at firm) * TFP_ratio * u_ratio * (1+m)."""
    carried = np.array([10.0, 20.0, 30.0, 40.0, 5.0])
    f = _offer(_setter(), carried)

    # Firm 0 holds individuals 0 and 1 -> mean rate 15.
    expected_firm0 = 15.0 * (TFP[0] / PREV_TFP[0]) * (EFFORT[0] / PREV_EFFORT[0]) * (1 + MARKUP[0])
    np.testing.assert_allclose(f(0, 1.0), expected_firm0)


def test__offer_ignores_previous_employee_income():
    """employee_income now holds earnings; reading it would re-import u."""
    carried = np.array([10.0, 20.0, 30.0, 40.0, 5.0])
    setter = _setter()
    baseline = setter.get_offered_wage_given_labour_inputs_function(
        corresponding_firm=CORRESPONDING_FIRM,
        current_individual_labour_inputs=np.array([1.0, 1.0, 1.0, 1.0, 0.0]),
        previous_employee_income=np.full(5, 1.0),
        current_target_production=np.full(N_FIRMS, 10.0),
        current_limiting_intermediate_inputs=np.full(N_FIRMS, 10.0),
        current_limiting_capital_inputs=np.full(N_FIRMS, 10.0),
        industry_labour_productivity_by_firm=np.full(N_FIRMS, 1.0),
        initial_wage_per_capita=INITIAL_WAGE,
        current_wage_per_capita=np.full(N_FIRMS, 55.0),
        current_labour_productivity_factor=EFFORT,
        prev_labour_productivity_factor=PREV_EFFORT,
        current_wage_tightness_markup=MARKUP,
        income_taxes=INCOME_TAX,
        employee_social_insurance_tax=EMPLOYEE_SI,
        employer_social_insurance_tax=EMPLOYER_SI,
        unemployment_benefits_by_individual=0.0,
        current_tfp_multiplier=TFP,
        prev_tfp_multiplier=PREV_TFP,
        carried_wage_rate=carried.copy(),
    )
    altered = _offer(_setter(), carried.copy())  # previous_employee_income = 999
    np.testing.assert_allclose(baseline(0, 1.0), altered(0, 1.0))


def test__offer_can_fall_when_effort_falls():
    """u_ratio is the only factor able to push an offer below its reference."""
    carried = np.array([10.0, 20.0, 30.0, 40.0, 5.0])
    falling_effort = np.array([0.5, 0.5, 0.5])
    f = _offer(
        _setter(),
        carried,
        markup=np.zeros(N_FIRMS),
        effort=falling_effort,
        prev_effort=np.ones(N_FIRMS),
        tfp=TFP,
        prev_tfp=TFP,
    )

    assert f(0, 1.0) < 15.0


def test__offer_is_non_decreasing_without_the_effort_ratio():
    """Documents why u is retained: (1+m) and TFP ratios cannot lower an offer."""
    carried = np.array([10.0, 20.0, 30.0, 40.0, 5.0])
    f = _offer(_setter(), carried, effort=np.ones(N_FIRMS), prev_effort=np.ones(N_FIRMS), tfp=TFP, prev_tfp=PREV_TFP)

    assert f(0, 1.0) >= 15.0


def test__firm_without_workforce_falls_back_to_initial_anchor():
    carried = np.array([10.0, 20.0, 30.0, 40.0, 5.0])
    # Nobody works at firm 2.
    lonely = np.array([0, 0, 1, 1, -1])
    f = _offer(
        _setter(),
        carried,
        markup=np.zeros(N_FIRMS),
        effort=np.ones(N_FIRMS),
        prev_effort=np.ones(N_FIRMS),
        tfp=TFP,
        prev_tfp=TFP,
        corresponding_firm=lonely,
    )

    np.testing.assert_allclose(f(2, 1.0), INITIAL_WAGE[2] / TAX_GROSS_UP)


def test__offer_is_floored_at_unemployment_benefits():
    carried = np.array([1e-9, 1e-9, 1e-9, 1e-9, 0.0])
    f = _offer(_setter(), carried, benefits=42.0)

    assert f(0, 1.0) == 42.0


def test__invalid_indexation_base_is_rejected():
    with pytest.raises(ValueError, match="indexation_base"):
        _setter(indexation_base="nonsense")


def test__realised_base_requires_the_ratio():
    setter = _setter(indexation_base="realised_productivity")
    carried = np.array([10.0, 20.0, 30.0, 40.0, 5.0])
    with pytest.raises(ValueError, match="realised_productivity_ratio"):
        _run(setter, carried)


def test__realised_base_indexes_to_productivity_not_tfp():
    """The whole point of U-A2b: TFP growth must not enter the rate."""
    setter = _setter(indexation_base="realised_productivity")
    carried = np.array([10.0, 20.0, 30.0, 40.0, 5.0])
    realised = np.array([1.01, 1.02, 1.03])

    _run(setter, carried, realised=realised)

    np.testing.assert_allclose(carried[:4], np.array([10.0, 20.0, 30.0, 40.0]) * realised[FIRM_OF_EMPLOYEE])


def test__tfp_base_ignores_the_realised_ratio():
    setter = _setter(indexation_base="tfp_multiplier")
    carried = np.array([10.0, 20.0, 30.0, 40.0, 5.0])

    _run(setter, carried, realised=np.array([5.0, 5.0, 5.0]))

    np.testing.assert_allclose(carried[:4], np.array([10.0, 20.0, 30.0, 40.0]) * (TFP / PREV_TFP)[FIRM_OF_EMPLOYEE])


def test__non_finite_realised_ratio_falls_back_to_no_growth():
    setter = _setter(indexation_base="realised_productivity")
    carried = np.array([10.0, 20.0, 30.0, 40.0, 5.0])

    _run(setter, carried, realised=np.array([np.nan, -1.0, np.inf]))

    np.testing.assert_allclose(carried[:4], np.array([10.0, 20.0, 30.0, 40.0]))
