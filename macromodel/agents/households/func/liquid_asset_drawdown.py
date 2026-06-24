"""Stage 5 Increment 1: shadow liquid-asset drawdown diagnostic."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LiquidAssetDrawdownResult:
    """Per-household liquid-asset drawdown diagnostic for one period."""

    funded_from_liquid_assets: np.ndarray
    residual_shortfall_after_lfa: np.ndarray


def compute_liquid_asset_drawdown(
    liquidity_shortfall: np.ndarray,
    available_lfa: np.ndarray,
) -> LiquidAssetDrawdownResult:
    """Compute the shadow liquid-asset repair margin for a liquidity shortfall.

    Invalid shortfalls are treated as no shortfall. Invalid or negative liquid
    assets are treated as no available liquid funding, leaving any valid
    positive shortfall as residual.
    """
    liquidity_shortfall = np.asarray(liquidity_shortfall, dtype=float)
    available_lfa = np.asarray(available_lfa, dtype=float)

    effective_shortfall = np.where(
        np.isfinite(liquidity_shortfall),
        np.maximum(liquidity_shortfall, 0.0),
        0.0,
    )
    effective_lfa = np.where(
        np.isfinite(available_lfa) & (available_lfa >= 0.0),
        available_lfa,
        0.0,
    )

    funded_from_liquid_assets = np.minimum(effective_shortfall, effective_lfa)
    residual_shortfall_after_lfa = effective_shortfall - funded_from_liquid_assets

    return LiquidAssetDrawdownResult(
        funded_from_liquid_assets=funded_from_liquid_assets,
        residual_shortfall_after_lfa=residual_shortfall_after_lfa,
    )
