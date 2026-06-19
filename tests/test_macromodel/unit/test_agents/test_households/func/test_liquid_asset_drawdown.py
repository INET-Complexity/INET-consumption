import numpy as np

from macromodel.agents.households.func.liquid_asset_drawdown import (
    LiquidAssetDrawdownResult,
    compute_liquid_asset_drawdown,
)


def _compute(liquidity_shortfall, available_lfa):
    return compute_liquid_asset_drawdown(
        liquidity_shortfall=np.asarray(liquidity_shortfall, dtype=float),
        available_lfa=np.asarray(available_lfa, dtype=float),
    )


def test__zero_and_negative_shortfall_have_no_drawdown_or_residual():
    result = _compute([0.0, -50.0], [100.0, 100.0])

    np.testing.assert_allclose(result.funded_from_liquid_assets, [0.0, 0.0])
    np.testing.assert_allclose(result.residual_shortfall_after_lfa, [0.0, 0.0])


def test__shortfall_below_liquid_assets_is_fully_funded():
    result = _compute([40.0], [100.0])

    np.testing.assert_allclose(result.funded_from_liquid_assets, [40.0])
    np.testing.assert_allclose(result.residual_shortfall_after_lfa, [0.0])


def test__shortfall_above_liquid_assets_leaves_residual():
    result = _compute([120.0], [50.0])

    np.testing.assert_allclose(result.funded_from_liquid_assets, [50.0])
    np.testing.assert_allclose(result.residual_shortfall_after_lfa, [70.0])


def test__zero_liquid_assets_leave_positive_shortfall_as_residual():
    result = _compute([120.0], [0.0])

    np.testing.assert_allclose(result.funded_from_liquid_assets, [0.0])
    np.testing.assert_allclose(result.residual_shortfall_after_lfa, [120.0])


def test__negative_liquid_assets_are_not_spendable():
    result = _compute([120.0], [-10.0])

    np.testing.assert_allclose(result.funded_from_liquid_assets, [0.0])
    np.testing.assert_allclose(result.residual_shortfall_after_lfa, [120.0])


def test__non_finite_liquid_assets_are_not_spendable():
    result = _compute([120.0, 130.0], [np.nan, np.inf])

    np.testing.assert_allclose(result.funded_from_liquid_assets, [0.0, 0.0])
    np.testing.assert_allclose(result.residual_shortfall_after_lfa, [120.0, 130.0])


def test__non_finite_shortfall_zeros_both_outputs():
    result = _compute([np.nan, np.inf], [100.0, 100.0])

    np.testing.assert_allclose(result.funded_from_liquid_assets, [0.0, 0.0])
    np.testing.assert_allclose(result.residual_shortfall_after_lfa, [0.0, 0.0])


def test__mixed_vector_preserves_accounting_identity():
    shortfall = np.asarray([100.0, 25.0, -5.0, np.nan, 10.0])
    result = _compute(shortfall, [40.0, 50.0, 50.0, 50.0, -5.0])

    effective_shortfall = np.asarray([100.0, 25.0, 0.0, 0.0, 10.0])
    np.testing.assert_allclose(
        result.funded_from_liquid_assets + result.residual_shortfall_after_lfa,
        effective_shortfall,
    )


def test__returns_liquid_asset_drawdown_result_dataclass():
    result = _compute([100.0], [40.0])

    assert isinstance(result, LiquidAssetDrawdownResult)
