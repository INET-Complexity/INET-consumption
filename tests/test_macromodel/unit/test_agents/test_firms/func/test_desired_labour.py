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

    def test__tfp_multiplier_reduces_desired_labour(self):
        """Fix A: desired labour must be divided by TFP so more productive firms hire fewer workers."""
        assert np.allclose(
            DefaultDesiredLabourSetter(
                consider_intermediate_inputs=False,
                consider_capital_inputs=False,
            ).compute_desired_labour(
                current_target_production=np.array([100.0, 200.0]),
                current_limiting_intermediate_inputs=np.array([100.0, 200.0]),
                current_limiting_capital_inputs=np.array([100.0, 200.0]),
                current_tfp_multiplier=np.array([2.0, 4.0]),
            ),
            np.array([50.0, 50.0]),
        )

    def test__tfp_multiplier_none_leaves_desired_labour_unchanged(self):
        """Fix A: omitting tfp_multiplier (None) must reproduce pre-TFP behaviour."""
        setter = DefaultDesiredLabourSetter(
            consider_intermediate_inputs=False,
            consider_capital_inputs=False,
        )
        result_no_tfp = setter.compute_desired_labour(
            current_target_production=np.array([100.0, 200.0]),
            current_limiting_intermediate_inputs=np.array([100.0, 200.0]),
            current_limiting_capital_inputs=np.array([100.0, 200.0]),
        )
        result_tfp1 = setter.compute_desired_labour(
            current_target_production=np.array([100.0, 200.0]),
            current_limiting_intermediate_inputs=np.array([100.0, 200.0]),
            current_limiting_capital_inputs=np.array([100.0, 200.0]),
            current_tfp_multiplier=np.array([1.0, 1.0]),
        )
        np.testing.assert_array_equal(result_no_tfp, result_tfp1)
