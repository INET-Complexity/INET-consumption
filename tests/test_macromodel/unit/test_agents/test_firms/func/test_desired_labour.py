import numpy as np

from macromodel.agents.firms.func.desired_labour import DefaultDesiredLabourSetter


class TestDesiredLabourSetter:
    def test__compute_desired_labour_without_input_constraints(self):
        assert np.allclose(
            DefaultDesiredLabourSetter(
                consider_intermediate_inputs=False,
                consider_capital_inputs=False,
            ).compute_desired_labour(
                current_target_production=np.array([100.0, 200.0]),
                current_limiting_intermediate_inputs=np.array([50.0, 300.0]),
                current_limiting_capital_inputs=np.array([300.0, 50.0]),
            ),
            np.array([100.0, 200.0]),
        )

    def test__compute_desired_labour_with_intermediate_constraints(self):
        assert np.allclose(
            DefaultDesiredLabourSetter(
                consider_intermediate_inputs=True,
                consider_capital_inputs=False,
            ).compute_desired_labour(
                current_target_production=np.array([100.0, 200.0]),
                current_limiting_intermediate_inputs=np.array([50.0, 300.0]),
                current_limiting_capital_inputs=np.array([300.0, 50.0]),
            ),
            np.array([50.0, 200.0]),
        )

    def test__compute_desired_labour_with_intermediate_and_capital_constraints(self):
        assert np.allclose(
            DefaultDesiredLabourSetter(
                consider_intermediate_inputs=True,
                consider_capital_inputs=True,
            ).compute_desired_labour(
                current_target_production=np.array([100.0, 200.0]),
                current_limiting_intermediate_inputs=np.array([50.0, 300.0]),
                current_limiting_capital_inputs=np.array([300.0, 150.0]),
            ),
            np.array([50.0, 150.0]),
        )
