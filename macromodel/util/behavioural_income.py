"""Component-aligned behavioural income; creates no household cash flows."""

import numpy as np


def reconcile_behavioural_income(
    *,
    non_property: np.ndarray,
    rental: np.ndarray,
    financial: np.ndarray,
    deposit_receipts: np.ndarray,
    borrower_interest: np.ndarray,
    imputed_rent: np.ndarray,
    subsistence: np.ndarray,
) -> dict[str, np.ndarray]:
    """Combine existing net nominal period flows, without HFCS tax replication.

    Retain raw operands and flag negative operands truncated to zero. A
    nonpositive final income uses nominal subsistence for log/ratio evaluation
    only. Positive income is unchanged. Neither adjustment books a receipt.
    """
    values = {
        "non_property_raw": np.asarray(non_property, dtype=float),
        "rental": np.asarray(rental, dtype=float),
        "financial": np.asarray(financial, dtype=float),
        "deposit_receipts": np.asarray(deposit_receipts, dtype=float),
        "borrower_interest": np.asarray(borrower_interest, dtype=float),
        "imputed_rent": np.asarray(imputed_rent, dtype=float),
    }
    shape = values["non_property_raw"].shape
    if len(shape) != 1 or any(v.shape != shape or not np.all(np.isfinite(v)) for v in values.values()):
        raise ValueError("Behavioural income components must be finite vectors with one value per household.")
    for name in ("deposit_receipts", "borrower_interest", "imputed_rent"):
        if np.any(values[name] < 0.0):
            raise ValueError(f"{name} must be nonnegative.")
    total_raw = values["non_property_raw"] + values["rental"] + values["financial"] + values["deposit_receipts"]
    n = np.maximum(values["non_property_raw"], 0.0)
    total = np.maximum(total_raw, 0.0)
    blend = np.sqrt(n) * np.sqrt(total)
    raw = blend - values["borrower_interest"] + values["imputed_rent"]
    floor_used = raw <= 0.0
    floor = np.asarray(subsistence, dtype=float)
    if floor.shape != shape or not np.all(np.isfinite(floor)) or np.any(floor < 0.0):
        raise ValueError("Subsistence income must be a finite nonnegative household vector.")
    if np.any(floor_used & (floor <= 0.0)):
        raise ValueError("Nonpositive behavioural income requires a positive subsistence fallback.")
    income = np.where(floor_used, floor, raw)
    if not np.all(np.isfinite(income)):
        raise ValueError("Reconciled behavioural income must be finite.")
    return {
        **values,
        "non_property": n,
        "total_raw": total_raw,
        "total": total,
        "blend": blend,
        "raw": raw,
        "component_floor_used": ((values["non_property_raw"] < 0.0) | (total_raw < 0.0)).astype(float),
        "floor_used": floor_used.astype(float),
        "income": income,
    }
