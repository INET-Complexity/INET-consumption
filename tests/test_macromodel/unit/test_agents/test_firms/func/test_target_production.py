import numpy as np

from macromodel.agents.firms.func.target_production import DefaultTargetProductionSetter


def test_target_production_subtracts_existing_inventory_and_adds_target_buffer():
    setter = DefaultTargetProductionSetter(
        existing_inventory_fraction=0.5,
        target_inventory_to_production_fraction=0.1,
        financial_constrains_fraction=0.0,
        maximum_debt_to_equity_ratio=2.0,
        intermediate_inputs_target_considers_labour_inputs=0.0,
        intermediate_inputs_target_considers_intermediate_inputs=0.0,
        intermediate_inputs_target_considers_capital_inputs=1.0,
        capital_inputs_target_considers_labour_inputs=0.0,
        capital_inputs_target_considers_intermediate_inputs=0.0,
        capital_inputs_target_considers_capital_inputs=1.0,
    )

    target_production = setter.compute_target_production(
        current_estimated_demand=np.array([100.0]),
        initial_inventory=np.array([0.0]),
        previous_inventory=np.array([20.0]),
        previous_production=np.array([100.0]),
        current_target_production=np.array([0.0]),
        current_limiting_intermediate_inputs=np.array([0.0]),
        current_limiting_capital_inputs=np.array([0.0]),
        current_firm_equity=np.array([100.0]),
        current_firm_debt=np.array([0.0]),
        previous_loans_applied_for=np.array([0.0]),
        current_firm_deposits=np.array([0.0]),
        interest_on_overdraft_rates=np.array([0.0]),
        interest_paid_on_loans=np.array([0.0]),
    )

    assert target_production == np.array([100.0])
