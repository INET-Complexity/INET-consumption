from typing import Literal

from pydantic import BaseModel, Field


class Consumption(BaseModel):
    name: Literal[
        "AutoregressiveGovernmentConsumptionSetter",
        "AutoregressiveGrowthGovernmentConsumptionSetter",
        "ConstantGrowthGovernmentConsumptionSetter",
        "ExpectedGrowthGovernmentConsumptionSetter",
        "ExogenousGovernmentConsumptionSetter",
    ] = "AutoregressiveGovernmentConsumptionSetter"
    path_name: str = "consumption"
    parameters: dict = Field(
        default_factory=lambda: {
            "consistency": 1.0,
            "sectoral_weights": "previous_desired",
        }
    )


class GovernmentFunctions(BaseModel):
    consumption: Consumption = Consumption()


class GovernmentEntitiesConfiguration(BaseModel):
    functions: GovernmentFunctions = GovernmentFunctions()
