"""Shared interest-rate transformations."""

from __future__ import annotations

import numpy as np


def compound_rate(rate: float | np.ndarray, periods: int) -> float | np.ndarray:
    """Compound a per-period decimal rate over ``periods`` periods."""
    if periods <= 0:
        raise ValueError("periods must be positive.")
    rate_array = np.asarray(rate, dtype=float)
    if np.any(rate_array <= -1.0):
        raise ValueError("rate must be greater than -1.")
    compounded = np.expm1(float(periods) * np.log1p(rate_array))
    return float(compounded) if compounded.shape == () else compounded


def fisher_real_rate(
    nominal_rate: float | np.ndarray,
    inflation_rate: float | np.ndarray,
) -> float | np.ndarray:
    """Return the exact Fisher real rate from same-frequency decimal rates."""
    nominal_array = np.asarray(nominal_rate, dtype=float)
    inflation_array = np.asarray(inflation_rate, dtype=float)
    if np.any(nominal_array <= -1.0):
        raise ValueError("nominal_rate must be greater than -1.")
    if np.any(inflation_array <= -1.0):
        raise ValueError("inflation_rate must be greater than -1.")
    real_rate = (1.0 + nominal_array) / (1.0 + inflation_array) - 1.0
    return float(real_rate) if real_rate.shape == () else real_rate
