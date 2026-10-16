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

    Retain raw operands and flag negative operands truncated to zero. Apply
    the same positive nominal subsistence lower bound to every final income:
    max(raw_income, subsistence). This is continuous at zero and at the bound;
    it cannot make evaluation income fall when raw income increases with a
    fixed bound. Keep raw income and flag every bound application. This floor
    affects consumption-rule inputs only and never books a cash receipt.
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
    floor = np.asarray(subsistence, dtype=float)
    if floor.shape != shape or not np.all(np.isfinite(floor)) or np.any(floor <= 0.0):
        raise ValueError("Subsistence income must be a finite positive household vector.")
    floor_used = raw < floor
    income = np.maximum(raw, floor)
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
