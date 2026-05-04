import warnings

import numpy as np

from macromodel.agents.firms.func.wage_setter import WorkEffortFirmWageSetter


class TestFirmWageSetter:
    def test__set_wages(self):
        WorkEffortFirmWageSetter(
            labour_market_tightness_markup_scale=0.1,
            markup_time_span=12,
            max_increase_in_work_effort=1.5,
        ).set_employee_income(
            corresponding_firm=np.array([0, 1]),
            current_individual_labour_inputs=np.array([1.0, 2.0]),
            current_individual_stating_new_job=np.array([True, False]),
            current_employee_income=np.array([10.0, 20.0]),
            current_individual_offered_wage=np.array([2.0, 3.0]),
            current_target_production=np.array([2.0, 3.0]),
            current_limiting_capital_inputs=np.array([2.0, 3.0]),
            current_limiting_intermediate_inputs=np.array([2.0, 3.0]),
            labour_inputs_from_employees=np.array([2.0, 3.0]),
            industry_labour_productivity_by_firm=np.array([2.0, 3.0]),
            estimated_ppi_inflation=1.0,
            employee_social_insurance_tax=0.1,
            employer_social_insurance_tax=0.1,
            income_taxes=0.3,
            initial_wage_per_capita=np.array([2.0, 3.0]),
            current_wage_per_capita=np.array([2.0, 3.0]),
            current_labour_productivity_factor=np.ones(2),
            prev_labour_productivity_factor=np.ones(2),
            current_wage_tightness_markup=np.zeros(2),
        )

    def test__offered_wage_uses_fallback_when_previous_productivity_is_zero(self):
        wage_setter = WorkEffortFirmWageSetter(
            labour_market_tightness_markup_scale=0.1,
            markup_time_span=12,
            max_increase_in_work_effort=1.5,
        )

        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")
            offered_wage_function = wage_setter.get_offered_wage_given_labour_inputs_function(
                corresponding_firm=np.array([0]),
                current_individual_labour_inputs=np.array([1.0]),
                previous_employee_income=np.array([10.0]),
                current_target_production=np.array([2.0]),
                current_limiting_capital_inputs=np.array([2.0]),
                current_limiting_intermediate_inputs=np.array([2.0]),
                industry_labour_productivity_by_firm=np.array([2.0]),
                initial_wage_per_capita=np.array([5.0]),
                current_wage_per_capita=np.array([5.0]),
                current_labour_productivity_factor=np.array([0.5]),
                prev_labour_productivity_factor=np.array([0.0]),
                current_wage_tightness_markup=np.array([0.2]),
                income_taxes=0.3,
                employee_social_insurance_tax=0.1,
                employer_social_insurance_tax=0.1,
                unemployment_benefits_by_individual=1.0,
                current_tfp_multiplier=np.array([1.0]),
            )

        assert recorded_warnings == []
        assert offered_wage_function(0, 1.0) == 3.0

    def test__tfp_multiplier_does_not_affect_set_employee_income(self):
        """Fix B regression: tfp_multiplier must not inflate employee wages."""
        setter = WorkEffortFirmWageSetter(
            labour_market_tightness_markup_scale=0.0,
            markup_time_span=12,
            max_increase_in_work_effort=1.5,
        )
        common_kwargs = dict(
            corresponding_firm=np.array([0, 0]),
            current_individual_labour_inputs=np.array([1.0, 1.0]),
            current_individual_stating_new_job=np.array([False, False]),
            current_employee_income=np.array([10.0, 10.0]),
            current_individual_offered_wage=np.array([10.0, 10.0]),
            current_target_production=np.array([2.0]),
            current_limiting_capital_inputs=np.array([1.0]),
            current_limiting_intermediate_inputs=np.array([1.0]),
            labour_inputs_from_employees=np.array([2.0]),
            industry_labour_productivity_by_firm=np.array([1.0]),
            estimated_ppi_inflation=0.0,
            employee_social_insurance_tax=0.0,
            employer_social_insurance_tax=0.0,
            income_taxes=0.0,
            initial_wage_per_capita=np.array([10.0]),
            current_wage_per_capita=np.array([10.0]),
            current_labour_productivity_factor=np.ones(1),
            prev_labour_productivity_factor=np.ones(1),
            current_wage_tightness_markup=np.zeros(1),
        )
        wages_no_tfp = setter.set_employee_income(**common_kwargs, current_tfp_multiplier=None)
        wages_with_tfp = setter.set_employee_income(**common_kwargs, current_tfp_multiplier=np.array([2.0]))
        np.testing.assert_array_equal(wages_no_tfp, wages_with_tfp)

    def test__tfp_multiplier_does_not_affect_offered_wage(self):
        """Fix B regression: tfp_multiplier must not inflate offered wages."""
        setter = WorkEffortFirmWageSetter(
            labour_market_tightness_markup_scale=0.0,
            markup_time_span=12,
            max_increase_in_work_effort=1.5,
        )
        common_kwargs = dict(
            corresponding_firm=np.array([0]),
            current_individual_labour_inputs=np.array([1.0]),
            previous_employee_income=np.array([10.0]),
            current_target_production=np.array([2.0]),
            current_limiting_capital_inputs=np.array([1.0]),
            current_limiting_intermediate_inputs=np.array([1.0]),
            industry_labour_productivity_by_firm=np.array([1.0]),
            initial_wage_per_capita=np.array([10.0]),
            current_wage_per_capita=np.array([10.0]),
            current_labour_productivity_factor=np.ones(1),
            prev_labour_productivity_factor=np.ones(1),
            current_wage_tightness_markup=np.zeros(1),
            income_taxes=0.0,
            employee_social_insurance_tax=0.0,
            employer_social_insurance_tax=0.0,
            unemployment_benefits_by_individual=0.0,
        )
        f_no_tfp = setter.get_offered_wage_given_labour_inputs_function(**common_kwargs, current_tfp_multiplier=None)
        f_with_tfp = setter.get_offered_wage_given_labour_inputs_function(**common_kwargs, current_tfp_multiplier=np.array([2.0]))
        assert f_no_tfp(0, 1.0) == f_with_tfp(0, 1.0)
