"""Household social transfer determination implementation.

This module implements social transfer allocation through:
- Transfer amount calculation
- Household-specific distribution
- Model-based predictions
- Equal allocation options

The implementation handles:
- Transfer budget allocation
- Household characteristics
- Model-driven predictions
- Distribution normalization
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np


def _validate_social_transfer_budget(n_households: int, total_other_social_transfers: float) -> float:
    """Validate the household count and transfer budget before allocation."""
    if n_households < 0:
        raise ValueError("Number of households must be non-negative.")
    try:
        budget = float(total_other_social_transfers)
    except (TypeError, ValueError) as error:
        raise ValueError("Social-transfer budget must be a finite non-negative number.") from error
    if not np.isfinite(budget) or budget < 0.0:
        raise ValueError("Social-transfer budget must be a finite non-negative number.")
    return budget


def _allocate_modelled_social_transfers(
    n_households: int,
    total_other_social_transfers: float,
    independents: np.ndarray,
    model: Optional[Any],
) -> np.ndarray:
    """Allocate a non-negative transfer budget from model predictions with safe fallbacks."""
    budget = _validate_social_transfer_budget(n_households, total_other_social_transfers)
    if n_households == 0 or budget == 0.0:
        return np.zeros(n_households, dtype=float)
    if model is None:
        raise ValueError("A social-transfer allocation model is required for a positive budget.")

    values = np.asarray(independents, dtype=float)
    if values.ndim != 2 or values.shape[0] != n_households:
        raise ValueError("Social-transfer allocation inputs must have one row per household.")
    values = np.where(np.isfinite(values), values, 0.0)
    column_totals = values.sum(axis=0, keepdims=True)
    normalized_values = np.divide(
        values,
        column_totals,
        out=np.zeros_like(values, dtype=float),
        where=column_totals != 0.0,
    )
    predicted_weights = np.asarray(model.predict(normalized_values), dtype=float).reshape(-1)
    if predicted_weights.shape != (n_households,):
        raise ValueError("Social-transfer allocation model must return one value per household.")
    predicted_weights = np.where(np.isfinite(predicted_weights) & (predicted_weights > 0.0), predicted_weights, 0.0)
    total_weight = predicted_weights.sum()
    if total_weight <= 0.0:
        return np.full(n_households, budget / n_households, dtype=float)
    return budget * predicted_weights / total_weight


class SocialTransfersSetter(ABC):
    """Abstract base class for social transfer allocation.

    Defines interface for determining transfer amounts based on:
    - Total transfer budget
    - Household characteristics
    - Model predictions
    - Distribution rules

    Attributes:
        independents (list[str]): Independent variables for transfer calculation
    """

    def __init__(self, independents: list[str]):
        self.independents = independents

    @abstractmethod
    def get_social_transfers(
        self,
        n_households: int,
        total_other_social_transfers: float,
        current_independents: np.ndarray,
        initial_independents: np.ndarray,
        model: Optional[Any],
    ) -> np.ndarray:
        """Calculate household social transfers.

        Args:
            n_households (int): Number of households
            total_other_social_transfers (float): Total transfer budget
            current_independents (np.ndarray): Current independent variables
            initial_independents (np.ndarray): Initial independent variables
            model (Optional[Any]): Prediction model

        Returns:
            np.ndarray: Transfer amounts by household
        """
        pass


class EqualSocialTransfersSetter(SocialTransfersSetter):
    """Simple transfer implementation using equal allocation.

    Distributes total transfer budget equally among all households.
    Used for scenarios where household-specific allocation is not needed.
    """

    def get_social_transfers(
        self,
        n_households: int,
        total_other_social_transfers: float,
        current_independents: np.ndarray,
        initial_independents: np.ndarray,
        model: Optional[Any],
    ) -> np.ndarray:
        """Return equal transfer amounts for all households.

        Args:
            n_households (int): Number of households
            total_other_social_transfers (float): Total transfer budget
            current_independents (np.ndarray): Current independent variables
            initial_independents (np.ndarray): Initial independent variables
            model (Optional[Any]): Prediction model

        Returns:
            np.ndarray: Uniform transfer amount array
        """
        budget = _validate_social_transfer_budget(n_households, total_other_social_transfers)
        if n_households == 0:
            return np.zeros(0, dtype=float)
        return np.full(n_households, budget / n_households)


class ConstantSocialTransfersSetter(SocialTransfersSetter):
    def get_social_transfers(
        self,
        n_households: int,
        total_other_social_transfers: float,
        current_independents: np.ndarray,
        initial_independents: np.ndarray,
        model: Optional[Any],
    ) -> np.ndarray:
        return _allocate_modelled_social_transfers(
            n_households=n_households,
            total_other_social_transfers=total_other_social_transfers,
            independents=initial_independents,
            model=model,
        )


class DefaultSocialTransfersSetter(SocialTransfersSetter):
    """Default implementation of social transfer allocation.

    Implements transfer determination through:
    - Model-based predictions
    - Variable normalization
    - Distribution adjustment
    """

    def get_social_transfers(
        self,
        n_households: int,
        total_other_social_transfers: float,
        current_independents: np.ndarray,
        initial_independents: np.ndarray,
        model: Optional[Any],
    ) -> np.ndarray:
        """Calculate transfers using default behavior.

        Determines transfers through:
        - Variable normalization
        - Model prediction
        - Budget allocation

        Args:
            n_households (int): Number of households
            total_other_social_transfers (float): Total transfer budget
            current_independents (np.ndarray): Current independent variables
            initial_independents (np.ndarray): Initial independent variables
            model (Optional[Any]): Prediction model

        Returns:
            np.ndarray: Transfer amounts by household
        """
        return _allocate_modelled_social_transfers(
            n_households=n_households,
            total_other_social_transfers=total_other_social_transfers,
            independents=current_independents,
            model=model,
        )
