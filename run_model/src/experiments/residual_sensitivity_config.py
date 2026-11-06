"""Isolated configuration overlays for the proposed residual sensitivity."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from macromodel.configurations import SimulationConfiguration


@dataclass(frozen=True)
class ResidualCase:
    model_seed: int
    policy: str
    draw_seed: int | None = None


def proposed_residual_cases() -> tuple[ResidualCase, ...]:
    """Return three anchored and nine crossed fixed-draw cases; never run them."""
    return tuple(
        case
        for seed in (12, 13, 14)
        for case in (
            ResidualCase(seed, "anchored"),
            *(ResidualCase(seed, "drawn", draw) for draw in (12001, 12002, 12003)),
        )
    )


def residual_case_configuration(base: SimulationConfiguration, case: ResidualCase) -> SimulationConfiguration:
    """Copy the effective config, changing only the declared policy and seeds."""
    if not isinstance(base, SimulationConfiguration) or not isinstance(case, ResidualCase):
        raise TypeError("Expected a SimulationConfiguration and ResidualCase.")
    for name, value in (("model_seed", case.model_seed), ("draw_seed", case.draw_seed)):
        if name == "draw_seed" and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value < 2**32:
            raise ValueError(f"{name} must be an integer in [0, 2**32).")
    if case.policy not in ("anchored", "drawn"):
        raise ValueError("policy must be anchored or drawn.")
    if (case.policy == "anchored") != (case.draw_seed is None):
        raise ValueError("Only drawn cases require a draw_seed.")
    if set(base.country_configurations) != {"FRA"}:
        raise ValueError("This design requires exactly the FRA country configuration.")
    function = base.country_configurations["FRA"].households.functions.consumption
    parameters = function.parameters
    if function.name != "CreditAugmentedConsumption":
        raise ValueError("The sensitivity requires CreditAugmentedConsumption.")
    if parameters.get("idiosyncratic_persistence") != "initial_state":
        raise ValueError("The base must retain the initial_state policy.")
    if parameters.get("idiosyncratic_seed") is not None:
        raise ValueError("The retained base must not have an explicit residual seed.")
    sigma = parameters.get("idiosyncratic_sd")
    if isinstance(sigma, bool) or not isinstance(sigma, (int, float)) or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("A finite positive fitted residual SD is required.")
    result = base.model_copy(deep=True)
    result.seed = int(case.model_seed)
    target = result.country_configurations["FRA"].households.functions.consumption.parameters
    if case.policy == "drawn":
        target["idiosyncratic_persistence"] = "fixed_effect"
        target["idiosyncratic_seed"] = int(case.draw_seed)
    return result
