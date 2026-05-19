from abc import ABC, abstractmethod

import numpy as np


class LabourProductivitySetter(ABC):
    """Abstract base class for determining labor productivity adjustments.

    This class defines strategies for calculating labor productivity factors
    based on:
    - Work effort potential and limits
    - Labour utilisation relative to production targets
    - Industry-specific productivity baselines
    - Adjustment speed parameters

    The productivity setting process considers:
    - Maximum allowable work effort increases
    - Under-utilisation of retained workers
    - Speed of productivity adjustments

    Attributes:
        min_labour_utilisation_factor (float): Minimum allowed labour
            productivity factor after work-effort adjustment
        max_labour_utilisation_factor (float): Maximum allowed increase in
            productivity through work effort
        consider_intermediate_inputs (float): Weight given to intermediate
            input constraints (deprecated, retained for config compatibility)
        consider_capital_inputs (float): Weight given to capital input
            constraints (deprecated, retained for config compatibility)
        work_effort_increase_speed (float): Rate at which work effort
            adjustments are implemented
    """

    def __init__(
        self,
        max_labour_utilisation_factor: float,
        consider_intermediate_inputs: float,
        consider_capital_inputs: float,
        work_effort_increase_speed: float,
        min_labour_utilisation_factor: float = 0.0,
    ) -> None:
        """Initialize the labor productivity setter with adjustment parameters.

        Args:
            max_labour_utilisation_factor (float): Maximum allowed increase in
                productivity through work effort
            consider_intermediate_inputs (float): Deprecated compatibility
                parameter; input constraints are handled by desired labour and
                production.
            consider_capital_inputs (float): Deprecated compatibility parameter;
                input constraints are handled by desired labour and production.
            work_effort_increase_speed (float): Rate of work effort adjustment
                implementation
            min_labour_utilisation_factor (float): Minimum allowed labour
                productivity factor after work-effort adjustment
        """
        if min_labour_utilisation_factor < 0.0 or max_labour_utilisation_factor < 0.0:
            raise ValueError("Labour productivity bounds must be non-negative.")
        if min_labour_utilisation_factor > 1.0:
            raise ValueError("min_labour_utilisation_factor cannot exceed 1.0.")
        if min_labour_utilisation_factor > max_labour_utilisation_factor:
            raise ValueError("min_labour_utilisation_factor cannot exceed max_labour_utilisation_factor.")

        self.min_labour_utilisation_factor = min_labour_utilisation_factor
        self.max_labour_utilisation_factor = max_labour_utilisation_factor
        self.consider_intermediate_inputs = max(0.0, min(1.0, consider_intermediate_inputs))
        self.consider_capital_inputs = max(0.0, min(1.0, consider_capital_inputs))
        self.work_effort_increase_speed = work_effort_increase_speed

    @abstractmethod
    def compute_labour_productivity_factor(
        self,
        current_target_production: np.ndarray,
        current_limiting_intermediate_inputs: np.ndarray,
        current_limiting_capital_inputs: np.ndarray,
        labour_inputs_from_employees: np.ndarray,
        industry_labour_productivity_by_firm: np.ndarray,
        current_tfp_multiplier: np.ndarray = None,
    ) -> np.ndarray:
        """Calculate labor productivity adjustment factors for each firm.

        Determines appropriate productivity multipliers considering:
        - Production targets
        - Current labour utilisation
        - Current labor inputs and industry standards
        - Maximum allowable adjustments

        Args:
            current_target_production (np.ndarray): Target production levels
            current_limiting_intermediate_inputs (np.ndarray): Deprecated and
                ignored. Input constraints belong in desired labour and
                production.
            current_limiting_capital_inputs (np.ndarray): Deprecated and
                ignored. Input constraints belong in desired labour and
                production.
            labour_inputs_from_employees (np.ndarray): Current labor input levels
            industry_labour_productivity_by_firm (np.ndarray): Industry standard
                productivity levels by firm
            current_tfp_multiplier (np.ndarray): Per-firm TFP multiplier. When
                provided, normal labour capacity is measured in TFP-adjusted
                effective units so technology gains are not misread as extra
                work effort.

        Returns:
            np.ndarray: Labor productivity adjustment factors by firm
        """
        pass


class WorkEffortLabourProductivitySetter(LabourProductivitySetter):
    """Implementation of productivity setting based on work effort.

    This class implements a strategy that:
    1. Compares target production with normal labour capacity
    2. Calculates labour utilisation or required work effort
    3. Limits increases to feasible ranges
    4. Implements changes at specified speed

    The approach ensures that:
    - Input constraints do not directly reduce labour utilisation
    - Adjustments stay within feasible bounds
    - Changes occur at appropriate speeds
    """

    def compute_labour_productivity_factor(
        self,
        current_target_production: np.ndarray,
        current_limiting_intermediate_inputs: np.ndarray,
        current_limiting_capital_inputs: np.ndarray,
        labour_inputs_from_employees: np.ndarray,
        industry_labour_productivity_by_firm: np.ndarray,
        current_tfp_multiplier: np.ndarray = None,
    ) -> np.ndarray:
        """Calculate productivity factors based on work effort adjustments.

        The method:
        1. Compares target production with normal labour capacity
        2. Calculates utilisation or required productivity increase
        3. Applies speed and maximum constraints

        Args:
            current_target_production (np.ndarray): Target production levels
            current_limiting_intermediate_inputs (np.ndarray): Intermediate
                input production capacity. Ignored by this calculation.
            current_limiting_capital_inputs (np.ndarray): Capital input
                production capacity. Ignored by this calculation.
            labour_inputs_from_employees (np.ndarray): Current labor inputs
            industry_labour_productivity_by_firm (np.ndarray): Industry
                standard productivity levels
            current_tfp_multiplier (np.ndarray): Per-firm TFP multiplier. If
                omitted, defaults to 1.0 for backward compatibility.

        Returns:
            np.ndarray: Productivity adjustment factors, where 1.0 represents
                normal utilisation, values below 1.0 represent under-utilised
                retained labour, and values above 1.0 represent increased work
                effort. The final result is clamped to the configured
                utilisation floor and ceiling.
        """
        tfp_multiplier = (
            current_tfp_multiplier
            if current_tfp_multiplier is not None
            else np.ones_like(current_target_production, dtype=float)
        )
        normal_labour_capacity = labour_inputs_from_employees * industry_labour_productivity_by_firm * tfp_multiplier
        required_productivity_factor = np.divide(
            current_target_production,
            normal_labour_capacity,
            out=np.ones_like(current_target_production, dtype=float),
            where=normal_labour_capacity > 0,
        )
        labour_productivity_factor = 1.0 + self.work_effort_increase_speed * (required_productivity_factor - 1.0)
        return np.clip(
            labour_productivity_factor,
            self.min_labour_utilisation_factor,
            self.max_labour_utilisation_factor,
        )
