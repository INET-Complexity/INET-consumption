import numpy as np

from macromodel.agents.firms.func.labour_productivity import WorkEffortLabourProductivitySetter


class TestWorkEffortLabourProductivitySetter:
    def test__normal_target_sets_normal_utilisation(self):
        labour_productivity_factor = WorkEffortLabourProductivitySetter(
            max_increase_in_work_effort=1.5,
            consider_intermediate_inputs=True,
            consider_capital_inputs=True,
            work_effort_increase_speed=1.0,
        ).compute_labour_productivity_factor(
            current_target_production=np.array([100.0]),
            current_limiting_intermediate_inputs=np.array([0.0]),
            current_limiting_capital_inputs=np.array([0.0]),
            labour_inputs_from_employees=np.array([10.0]),
            industry_labour_productivity_by_firm=np.array([10.0]),
        )

        assert np.allclose(labour_productivity_factor, np.array([1.0]))

    def test__high_target_increases_utilisation_up_to_cap(self):
        labour_productivity_factor = WorkEffortLabourProductivitySetter(
            max_increase_in_work_effort=1.5,
            consider_intermediate_inputs=True,
            consider_capital_inputs=True,
            work_effort_increase_speed=1.0,
        ).compute_labour_productivity_factor(
            current_target_production=np.array([120.0, 200.0]),
            current_limiting_intermediate_inputs=np.array([120.0, 200.0]),
            current_limiting_capital_inputs=np.array([120.0, 200.0]),
            labour_inputs_from_employees=np.array([10.0, 10.0]),
            industry_labour_productivity_by_firm=np.array([10.0, 10.0]),
        )

        assert np.allclose(labour_productivity_factor, np.array([1.2, 1.5]))

    def test__low_target_reduces_utilisation(self):
        labour_productivity_factor = WorkEffortLabourProductivitySetter(
            max_increase_in_work_effort=1.5,
            consider_intermediate_inputs=True,
            consider_capital_inputs=True,
            work_effort_increase_speed=1.0,
        ).compute_labour_productivity_factor(
            current_target_production=np.array([70.0]),
            current_limiting_intermediate_inputs=np.array([70.0]),
            current_limiting_capital_inputs=np.array([70.0]),
            labour_inputs_from_employees=np.array([10.0]),
            industry_labour_productivity_by_firm=np.array([10.0]),
        )

        assert np.allclose(labour_productivity_factor, np.array([0.7]))

    def test__zero_intermediate_inputs_do_not_reduce_utilisation(self):
        labour_productivity_factor = WorkEffortLabourProductivitySetter(
            max_increase_in_work_effort=1.5,
            consider_intermediate_inputs=True,
            consider_capital_inputs=True,
            work_effort_increase_speed=1.0,
        ).compute_labour_productivity_factor(
            current_target_production=np.array([100.0]),
            current_limiting_intermediate_inputs=np.array([0.0]),
            current_limiting_capital_inputs=np.array([0.0]),
            labour_inputs_from_employees=np.array([10.0]),
            industry_labour_productivity_by_firm=np.array([10.0]),
        )

        assert np.allclose(labour_productivity_factor, np.array([1.0]))

    def test__zero_labour_capacity_returns_normal_utilisation(self):
        labour_productivity_factor = WorkEffortLabourProductivitySetter(
            max_increase_in_work_effort=1.5,
            consider_intermediate_inputs=True,
            consider_capital_inputs=True,
            work_effort_increase_speed=1.0,
        ).compute_labour_productivity_factor(
            current_target_production=np.array([100.0]),
            current_limiting_intermediate_inputs=np.array([100.0]),
            current_limiting_capital_inputs=np.array([100.0]),
            labour_inputs_from_employees=np.array([0.0]),
            industry_labour_productivity_by_firm=np.array([10.0]),
        )

        assert np.isfinite(labour_productivity_factor).all()
        assert np.allclose(labour_productivity_factor, np.array([1.0]))
