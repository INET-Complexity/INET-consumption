import numpy as np
import pytest

from macromodel.agents.households.func.post_liquidation_settlement import (
    PostLiquidationSettlement,
    settle_post_liquidation,
)


def _settle(**overrides):
    values = {
        "base_lfa": np.asarray([100.0, 20.0, 5.0]),
        "base_ifa": np.asarray([50.0, 5.0, 0.0]),
        "planned_liquidation_total": np.asarray([8.0, 5.0, 0.0]),
        "residual_shortfall_after_granted_credit": np.asarray([0.0, 4.0, 0.0]),
    }
    values.update(overrides)
    return settle_post_liquidation(**values)


def test__settles_liquidation_into_lfa_and_ifa_once():
    result = _settle()

    np.testing.assert_allclose(result.post_liquidation_lfa, [108.0, 25.0, 5.0])
    np.testing.assert_allclose(result.post_liquidation_ifa, [42.0, 0.0, 0.0])
    np.testing.assert_allclose(result.settled_liquidation_total, [8.0, 5.0, 0.0])
    np.testing.assert_allclose(result.residual_shortfall_after_settlement, [0.0, 4.0, 0.0])


def test__zero_liquidation_preserves_financial_bases():
    result = _settle(planned_liquidation_total=np.zeros(3))

    np.testing.assert_allclose(result.post_liquidation_lfa, [100.0, 20.0, 5.0])
    np.testing.assert_allclose(result.post_liquidation_ifa, [50.0, 5.0, 0.0])
    np.testing.assert_allclose(result.residual_shortfall_after_settlement, [0.0, 4.0, 0.0])


def test__unhonoured_reserved_liquidation_fails_settlement():
    with pytest.raises(RuntimeError, match="cannot be honoured"):
        _settle(
            base_ifa=np.asarray([2.0, 3.0, 0.0]),
            planned_liquidation_total=np.asarray([8.0, 5.0, 0.0]),
            residual_shortfall_after_granted_credit=np.asarray([1.0, 4.0, 0.0]),
        )


def test__financial_asset_conservation_holds():
    result = _settle()

    np.testing.assert_allclose(
        result.post_liquidation_lfa + result.post_liquidation_ifa,
        np.asarray([150.0, 25.0, 5.0]),
    )


def test__rejects_non_finite_or_mismatched_inputs():
    with pytest.raises(RuntimeError, match="finite inputs"):
        _settle(base_lfa=np.asarray([np.nan, 20.0, 5.0]))

    with pytest.raises(ValueError, match="equal-length vectors"):
        _settle(base_ifa=np.asarray([50.0, 3.0]))


def test__returns_immutable_settlement_carrier():
    assert isinstance(_settle(), PostLiquidationSettlement)
