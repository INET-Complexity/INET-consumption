from macromodel.agents.households.func.consumption import CreditAugmentedConsumption
from macromodel.configurations.households_configuration import (
    ConsumptionFunction,
    HouseholdsConfiguration,
    HouseholdsFunctions,
)
from macromodel.util.function_mapping import functions_from_model


def test_households_configuration_accepts_credit_augmented_consumption():
    configuration = HouseholdsConfiguration(
        functions=HouseholdsFunctions(
            consumption=ConsumptionFunction(name="CreditAugmentedConsumption"),
        )
    )

    assert configuration.functions.consumption.name == "CreditAugmentedConsumption"


def test_functions_from_model_instantiates_credit_augmented_consumption():
    configuration = HouseholdsConfiguration(
        functions=HouseholdsFunctions(
            consumption=ConsumptionFunction(
                name="CreditAugmentedConsumption",
                parameters={
                    "consumption_smoothing_fraction": 0.0,
                    "consumption_smoothing_window": 1,
                    "minimum_consumption_fraction": 0.0,
                },
            )
        )
    )

    functions = functions_from_model(configuration.functions, loc="macromodel.agents.households")

    assert isinstance(functions["consumption"], CreditAugmentedConsumption)
