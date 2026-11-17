import numpy as np
import pytest

from macromodel.agents.firms.func.labour_productivity import WorkEffortLabourProductivitySetter

DEFAULT_MIN = 0.9
DEFAULT_MAX = 1.1
DEFAULT_SPEED = 1.0


def make_setter(
    min_labour_utilisation_factor: float = DEFAULT_MIN,
    max_labour_utilisation_factor: float = DEFAULT_MAX,
    work_effort_increase_speed: float = DEFAULT_SPEED,
    consider_intermediate_inputs: float = 1.0,
    consider_capital_inputs: float = 1.0,
):
    return WorkEffortLabourProductivitySetter(
        min_labour_utilisation_factor=min_labour_utilisation_factor,
        max_labour_utilisation_factor=max_labour_utilisation_factor,
        consider_intermediate_inputs=consider_intermediate_inputs,
        consider_capital_inputs=consider_capital_inputs,
        work_effort_increase_speed=work_effort_increase_speed,
    )


class TestWorkEffortLabourProductivitySetter:
    def test__normal_target_sets_normal_utilisation(self):
        labour_productivity_factor = make_setter().compute_labour_productivity_factor(
            current_target_production=np.array([100.0]),
            current_limiting_intermediate_inputs=np.array([100.0]),
            current_limiting_capital_inputs=np.array([100.0]),
            labour_inputs_from_employees=np.array([10.0]),
            industry_labour_productivity_by_firm=np.array([10.0]),
        )

        assert np.allclose(labour_productivity_factor, np.array([1.0]))

    def test__high_target_increases_utilisation_up_to_cap(self):
        labour_productivity_factor = make_setter().compute_labour_productivity_factor(
            current_target_production=np.array([120.0, 200.0]),
            current_limiting_intermediate_inputs=np.array([120.0, 200.0]),
            current_limiting_capital_inputs=np.array([120.0, 200.0]),
            labour_inputs_from_employees=np.array([10.0, 10.0]),
            industry_labour_productivity_by_firm=np.array([10.0, 10.0]),
        )

        assert np.allclose(labour_productivity_factor, np.array([1.1, 1.1]))

    def test__tfp_adjusted_capacity_does_not_look_like_extra_effort(self):
        labour_productivity_factor = make_setter().compute_labour_productivity_factor(
            current_target_production=np.array([100.0]),
            current_limiting_intermediate_inputs=np.array([100.0]),
            current_limiting_capital_inputs=np.array([100.0]),
            labour_inputs_from_employees=np.array([5.0]),
            industry_labour_productivity_by_firm=np.array([10.0]),
            current_tfp_multiplier=np.array([2.0]),
        )

        assert np.allclose(labour_productivity_factor, np.array([1.0]))

    def test__low_target_floors_utilisation_at_min(self):
        labour_productivity_factor = make_setter().compute_labour_productivity_factor(
            current_target_production=np.array([70.0]),
            current_limiting_intermediate_inputs=np.array([70.0]),
            current_limiting_capital_inputs=np.array([70.0]),
            labour_inputs_from_employees=np.array([10.0]),
            industry_labour_productivity_by_firm=np.array([10.0]),
        )

        assert np.allclose(labour_productivity_factor, np.array([0.9]))

    def test__speed_below_one_still_respects_floor(self):
        labour_productivity_factor = make_setter(work_effort_increase_speed=0.5).compute_labour_productivity_factor(
            current_target_production=np.array([50.0]),
            current_limiting_intermediate_inputs=np.array([50.0]),
            current_limiting_capital_inputs=np.array([50.0]),
            labour_inputs_from_employees=np.array([10.0]),
            industry_labour_productivity_by_firm=np.array([10.0]),
        )

        assert np.allclose(labour_productivity_factor, np.array([0.9]))

    def test__zero_floor_preserves_previous_behavior(self):
        labour_productivity_factor = make_setter(
            min_labour_utilisation_factor=0.0, max_labour_utilisation_factor=1.5
        ).compute_labour_productivity_factor(
            current_target_production=np.array([70.0]),
            current_limiting_intermediate_inputs=np.array([70.0]),
            current_limiting_capital_inputs=np.array([70.0]),
            labour_inputs_from_employees=np.array([10.0]),
            industry_labour_productivity_by_firm=np.array([10.0]),
        )

        assert np.allclose(labour_productivity_factor, np.array([0.7]))

    def test__capital_limit_caps_effort(self):
        # Target needs 1.08 effort, capital allows only 100 = normal capacity.
        labour_productivity_factor = make_setter().compute_labour_productivity_factor(
            current_target_production=np.array([108.0]),
            current_limiting_intermediate_inputs=np.array([200.0]),
            current_limiting_capital_inputs=np.array([100.0]),
            labour_inputs_from_employees=np.array([10.0]),
            industry_labour_productivity_by_firm=np.array([10.0]),
        )

        assert np.allclose(labour_productivity_factor, np.array([1.0]))

    def test__intermediate_limit_caps_effort(self):
        labour_productivity_factor = make_setter().compute_labour_productivity_factor(
            current_target_production=np.array([108.0]),
            current_limiting_intermediate_inputs=np.array([95.0]),
            current_limiting_capital_inputs=np.array([200.0]),
            labour_inputs_from_employees=np.array([10.0]),
            industry_labour_productivity_by_firm=np.array([10.0]),
        )

        assert np.allclose(labour_productivity_factor, np.array([0.95]))

    def test__missing_inputs_lower_effort_only_to_floor(self):
        labour_productivity_factor = make_setter().compute_labour_productivity_factor(
            current_target_production=np.array([100.0]),
            current_limiting_intermediate_inputs=np.array([0.0]),
            current_limiting_capital_inputs=np.array([0.0]),
            labour_inputs_from_employees=np.array([10.0]),
            industry_labour_productivity_by_firm=np.array([10.0]),
        )

        assert np.allclose(labour_productivity_factor, np.array([DEFAULT_MIN]))

    def test__zero_weights_ignore_input_limits(self):
        labour_productivity_factor = make_setter(
            consider_intermediate_inputs=0.0, consider_capital_inputs=0.0
        ).compute_labour_productivity_factor(
            current_target_production=np.array([108.0]),
            current_limiting_intermediate_inputs=np.array([0.0]),
            current_limiting_capital_inputs=np.array([0.0]),
            labour_inputs_from_employees=np.array([10.0]),
            industry_labour_productivity_by_firm=np.array([10.0]),
        )

        assert np.allclose(labour_productivity_factor, np.array([1.08]))

    def test__input_limits_are_compared_in_tfp_effective_units(self):
        # Capacity 5 * 10 * 2 = 100; capital allows 100, target 108 -> effort 1.0.
        labour_productivity_factor = make_setter().compute_labour_productivity_factor(
            current_target_production=np.array([108.0]),
            current_limiting_intermediate_inputs=np.array([200.0]),
            current_limiting_capital_inputs=np.array([100.0]),
            labour_inputs_from_employees=np.array([5.0]),
            industry_labour_productivity_by_firm=np.array([10.0]),
            current_tfp_multiplier=np.array([2.0]),
        )

        assert np.allclose(labour_productivity_factor, np.array([1.0]))

    def test__zero_labour_capacity_returns_normal_utilisation(self):
        labour_productivity_factor = make_setter().compute_labour_productivity_factor(
            current_target_production=np.array([100.0]),
            current_limiting_intermediate_inputs=np.array([100.0]),
            current_limiting_capital_inputs=np.array([100.0]),
            labour_inputs_from_employees=np.array([0.0]),
            industry_labour_productivity_by_firm=np.array([10.0]),
        )

        assert np.isfinite(labour_productivity_factor).all()
        assert np.allclose(labour_productivity_factor, np.array([1.0]))

    @pytest.mark.parametrize(
        "min_labour_utilisation_factor,max_labour_utilisation_factor",
        [
            (-0.1, 1.1),
            (0.0, -0.1),
            (1.1, 1.2),
            (1.2, 1.1),
        ],
    )
    def test__invalid_bounds_raise_value_error(self, min_labour_utilisation_factor, max_labour_utilisation_factor):
        with pytest.raises(ValueError):
            make_setter(
                min_labour_utilisation_factor=min_labour_utilisation_factor,
                max_labour_utilisation_factor=max_labour_utilisation_factor,
            )
