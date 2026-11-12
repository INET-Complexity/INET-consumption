from abc import ABC, abstractmethod

import numpy as np


class DesiredLabourSetter(ABC):
    """Abstract base class for determining firms' desired labor inputs.

    This class defines strategies for calculating optimal labor demand based on:
    - Production targets
    - Input constraints (intermediate and capital)
    - Complementarity between labor and other inputs

    The labor demand calculation considers the interaction between different
    factors of production, with configurable weights for:
    - Intermediate input constraints (materials, supplies)
    - Capital input constraints (machinery, equipment)

    Attributes:
        consider_intermediate_inputs (float): Weight given to intermediate input constraints
            Clipped to range [0,1], where higher values mean stronger complementarity
        consider_capital_inputs (float): Weight given to capital input constraints
            Clipped to range [0,1], where higher values mean stronger complementarity
    """

    def __init__(
        self,
        consider_intermediate_inputs: float,
        consider_capital_inputs: float,
    ) -> None:
        """Initialize the desired labor setter with input consideration weights.

        Args:
            consider_intermediate_inputs (float): Weight for intermediate input constraints
                Will be clipped to range [0,1]
            consider_capital_inputs (float): Weight for capital input constraints
                Will be clipped to range [0,1]
        """
        self.consider_intermediate_inputs = max(0.0, min(1.0, consider_intermediate_inputs))
        self.consider_capital_inputs = max(0.0, min(1.0, consider_capital_inputs))

    @abstractmethod
    def compute_desired_labour(
        self,
        current_target_production: np.ndarray,
        current_limiting_intermediate_inputs: np.ndarray,
        current_limiting_capital_inputs: np.ndarray,
        current_tfp_multiplier: np.ndarray = None,
    ) -> np.ndarray:
        """Calculate desired labor inputs for each firm.

        Args:
            current_target_production (np.ndarray): Target production levels
            current_limiting_intermediate_inputs (np.ndarray): Production constraints
                from intermediate input availability
            current_limiting_capital_inputs (np.ndarray): Production constraints
                from capital input availability
            current_tfp_multiplier (np.ndarray): Per-firm TFP level. When provided,
                desired labour is divided by TFP so that more productive firms hire
                proportionally fewer workers for the same output target.

        Returns:
            np.ndarray: Desired labor inputs for each firm
        """
        pass


class DefaultDesiredLabourSetter(DesiredLabourSetter):
    """Default implementation of desired labor calculation.

    This class implements a labor demand strategy that:
    1. Starts with target production levels
    2. Adjusts for intermediate input constraints
    3. Further adjusts for capital input constraints

    The adjustments reflect the complementarity between labor and other
    inputs in the production process, ensuring labor demand is consistent
    with available complementary factors of production.
    """

    def compute_desired_labour(
        self,
        current_target_production: np.ndarray,
        current_limiting_intermediate_inputs: np.ndarray,
        current_limiting_capital_inputs: np.ndarray,
        current_tfp_multiplier: np.ndarray = None,
    ) -> np.ndarray:
        """Calculate desired labor using the default adjustment strategy.

        Applies intermediate/capital limits in output units, then converts the
        resulting output plan into pre-TFP labour units. This matches the
        Leontief production function with labour-augmenting TFP:
        Y = min(target, A * labour, intermediate_limit, capital_limit).

        Args:
            current_target_production (np.ndarray): Initial production targets
            current_limiting_intermediate_inputs (np.ndarray): Intermediate input constraints
            current_limiting_capital_inputs (np.ndarray): Capital input constraints
            current_tfp_multiplier (np.ndarray): Per-firm TFP level (optional).

        Returns:
            np.ndarray: Adjusted labor demand accounting for input complementarities and TFP
        """
        planned_output = np.asarray(current_target_production, dtype=float).copy()
        planned_output = np.minimum(
            planned_output,
            planned_output
            + self.consider_intermediate_inputs * (current_limiting_intermediate_inputs - planned_output),
        )
        planned_output = np.minimum(
            planned_output,
            planned_output + self.consider_capital_inputs * (current_limiting_capital_inputs - planned_output),
        )
        if current_tfp_multiplier is None:
            return planned_output
        return np.divide(
            planned_output,
            current_tfp_multiplier,
            out=planned_output.copy(),
            where=current_tfp_multiplier > 0,
        )
