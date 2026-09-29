"""Validation and normalization for household housing-flow observations."""

from __future__ import annotations

from typing import Optional

import numpy as np


def normalise_housing_flow(
    value: np.ndarray,
    *,
    name: str,
    expected_shape: Optional[tuple[int, ...]] = None,
) -> np.ndarray:
    """Validate one household housing-flow vector and apply the established zero floor."""
    flow = np.asarray(value, dtype=float)
    if flow.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional household vector.")
    if expected_shape is not None and flow.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {flow.shape}.")
    if not np.all(np.isfinite(flow)):
        raise ValueError(f"{name} must be finite.")
    return np.maximum(flow, 0.0)


def normalise_housing_flows(
    rent: np.ndarray,
    rent_imputed: np.ndarray,
    *,
    expected_shape: Optional[tuple[int, ...]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and normalize actual and imputed rent on one common boundary."""
    normalized_rent = normalise_housing_flow(rent, name="rent", expected_shape=expected_shape)
    normalized_imputed = normalise_housing_flow(
        rent_imputed,
        name="rent_imputed",
        expected_shape=expected_shape or normalized_rent.shape,
    )
    if normalized_rent.shape != normalized_imputed.shape:
        raise ValueError("rent and rent_imputed must have matching household shapes.")
    return normalized_rent, normalized_imputed
