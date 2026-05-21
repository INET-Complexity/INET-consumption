import numpy as np

from macromodel.agents.firms.func.demand_estimator import DefaultDemandEstimator


class TestDemandEstimator:
    def test__growth_adjustment_speeds_are_clipped_to_unit_interval(self):
        estimator = DefaultDemandEstimator(
            sectoral_growth_adjustment_speed=2.0,
            firm_growth_adjustment_speed=-1.0,
        )

        assert estimator.sectoral_growth_adjustment_speed == 1.0
        assert estimator.firm_growth_adjustment_speed == 0.0

    def test__compute_estimated_demand_uses_clipped_growth_adjustment_speeds(self):
        assert np.allclose(
            DefaultDemandEstimator(
                sectoral_growth_adjustment_speed=2.0,
                firm_growth_adjustment_speed=2.0,
            ).compute_estimated_demand(
                previous_demand=np.array([100.0]),
                estimated_growth_by_firm=np.array([0.5]),
                current_estimated_growth=0.1,
            ),
            np.array([165.0]),
        )

    def test__compute_estimated_demand(self):
        assert np.allclose(
            DefaultDemandEstimator(
                sectoral_growth_adjustment_speed=1.0,
                firm_growth_adjustment_speed=0.0,
            ).compute_estimated_demand(
                previous_demand=np.array([1.0, 2.0]),
                estimated_growth_by_firm=np.array([0.0, 0.1]),
                current_estimated_growth=0.0,
            ),
            np.array([1.0, 2.0]),
        )
