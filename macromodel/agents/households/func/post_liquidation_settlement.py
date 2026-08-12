"""Settle Stage 5 forced illiquid-asset liquidation against wealth bases."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PostLiquidationSettlement:
    """Authoritative post-liquidation financial-asset bases."""

    post_liquidation_lfa: np.ndarray
    post_liquidation_ifa: np.ndarray
    settled_liquidation_total: np.ndarray
    residual_shortfall_after_settlement: np.ndarray


def settle_post_liquidation(
    *,
    base_lfa: np.ndarray,
    base_ifa: np.ndarray,
    planned_liquidation_total: np.ndarray,
    residual_shortfall_after_granted_credit: np.ndarray,
) -> PostLiquidationSettlement:
    """Settle sanctioned liquidation once against final wealth bases.

    ``planned_liquidation_total`` is the reservation made before floor and goods
    demand were fixed. It must be executable in full at settlement; a shortfall
    is a failed transaction, not a reason to silently shrink the reservation.
    """
    arrays = {
        "base_lfa": np.asarray(base_lfa, dtype=float),
        "base_ifa": np.asarray(base_ifa, dtype=float),
        "planned_liquidation_total": np.asarray(planned_liquidation_total, dtype=float),
        "residual_shortfall_after_granted_credit": np.asarray(residual_shortfall_after_granted_credit, dtype=float),
    }
    shapes = {name: value.shape for name, value in arrays.items()}
    if len(set(shapes.values())) != 1 or next(iter(shapes.values()), ()) == ():
        raise ValueError(f"Stage 5 liquidation inputs must be equal-length vectors; got {shapes}.")
    if not all(np.all(np.isfinite(value)) for value in arrays.values()):
        raise RuntimeError("Stage 5 liquidation settlement requires finite inputs.")

    base_lfa_values = arrays["base_lfa"]
    base_ifa_values = arrays["base_ifa"]
    planned = np.maximum(arrays["planned_liquidation_total"], 0.0)
    residual_after_granted_credit = np.maximum(arrays["residual_shortfall_after_granted_credit"], 0.0)
    if np.any(base_ifa_values < 0.0):
        raise RuntimeError("Stage 5 liquidation settlement requires non-negative IFA bases.")
    if np.any(planned > base_ifa_values):
        raise RuntimeError("Reserved Stage 5 liquidation cannot be honoured at settlement.")
    settled = planned

    return PostLiquidationSettlement(
        post_liquidation_lfa=base_lfa_values + settled,
        post_liquidation_ifa=base_ifa_values - settled,
        settled_liquidation_total=settled,
        residual_shortfall_after_settlement=residual_after_granted_credit.copy(),
    )
