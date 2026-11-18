"""Rent/value floor shared by data construction and the model's rent rule (issue #197)."""

import numpy as np


def rent_floor(minimum_annual_rental_yield: float, property_value: np.ndarray, time_unit: int) -> np.ndarray:
    """Per-period rent floor phi * (time_unit / 12) * value, with phi an annual rent/value yield."""
    return minimum_annual_rental_yield * (time_unit / 12.0) * np.asarray(property_value, dtype=float)
