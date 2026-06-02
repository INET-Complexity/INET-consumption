import numpy as np

from macromodel.agents.firms.func.target_production import DefaultTargetProductionSetter


def _setter(**overrides):
    parameters = {
        "target_inventory_to_demand_fraction": 0.1,
        "financial_constrains_fraction": 0.0,
        "maximum_debt_to_equity_ratio": 2.0,
        "intermediate_inputs_target_considers_labour_inputs": 0.0,
        "intermediate_inputs_target_considers_intermediate_inputs": 0.0,
        "intermediate_inputs_target_considers_capital_inputs": 1.0,
        "capital_inputs_target_considers_labour_inputs": 0.0,
        "capital_inputs_target_considers_intermediate_inputs": 0.0,
        "capital_inputs_target_considers_capital_inputs": 1.0,
    }
    parameters.update(overrides)
    return DefaultTargetProductionSetter(**parameters)


def _compute(setter, **overrides):
    inputs = {
        "current_estimated_demand": np.array([100.0]),
        "initial_inventory": np.array([0.0]),
        "previous_inventory": np.array([20.0]),
        "previous_production": np.array([100.0]),
        "current_target_production": np.array([0.0]),
        "current_limiting_intermediate_inputs": np.array([0.0]),
        "current_limiting_capital_inputs": np.array([0.0]),
        "current_firm_equity": np.array([100.0]),
        "current_firm_debt": np.array([0.0]),
        "previous_loans_applied_for": np.array([0.0]),
        "current_firm_deposits": np.array([0.0]),
        "interest_on_overdraft_rates": np.array([0.0]),
        "interest_paid_on_loans": np.array([0.0]),
    }
    inputs.update(overrides)
    return setter.compute_target_production(**inputs)


def test_target_production_default_speed_uses_demand_linked_inventory_net_formula():
    target_production = _compute(_setter())

    assert np.allclose(target_production, np.array([90.0]))


def test_target_production_inventory_adjustment_speed_partially_closes_inventory_gap():
    target_production = _compute(_setter(inventory_adjustment_speed=0.25))

    assert np.allclose(target_production, np.array([97.5]))


def test_target_production_zero_inventory_adjustment_speed_uses_estimated_demand():
    target_production = _compute(_setter(inventory_adjustment_speed=0.0))

    assert np.allclose(target_production, np.array([100.0]))


def test_target_production_inventory_adjustment_speed_is_clipped():
    low_speed_target = _compute(_setter(inventory_adjustment_speed=-0.25))
    high_speed_target = _compute(_setter(inventory_adjustment_speed=1.25))

    assert np.allclose(low_speed_target, np.array([100.0]))
    assert np.allclose(high_speed_target, np.array([90.0]))


def test_target_production_is_floored_when_inventory_exceeds_demand_and_buffer():
    target_production = _compute(
        _setter(),
        current_estimated_demand=np.array([5.0]),
        previous_inventory=np.array([20.0]),
    )

    assert np.allclose(target_production, np.array([1e-12]))


def test_target_production_financial_constraint_penalty_applies_after_inventory_net_formula():
    target_production = _compute(
        _setter(financial_constrains_fraction=0.5, inventory_adjustment_speed=0.25),
        previous_loans_applied_for=np.array([10.0]),
        current_firm_equity=np.array([100.0]),
        current_firm_debt=np.array([0.0]),
    )

    assert np.allclose(target_production, np.array([95.0]))
